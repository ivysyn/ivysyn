#include <fstream>
#include <llvm-11/llvm/ADT/APFloat.h>
#include "InjectFuzzer.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

const std::string TENSORFLOW_PATH = "/home/ivyusr/ivysyn/src/frameworks/tensorflow-2.6-ivysyn/";
const std::string KERNEL_DIR = TENSORFLOW_PATH + "tensorflow/core/kernels/";
const std::string TF_IVYSYN_PATH = "/home/ivyusr/ivysyn/src/ivysyn/tensorflow/";
const std::string ONE_TO_ONE_FILE = TF_IVYSYN_PATH + "one_to_one_kernels.txt";

using namespace clang;
using namespace ast_matchers;

std::string get_source_filename(const SourceManager& SrcMgr, SourceLocation SrcLoc)
{
  const FileEntry* Entry = SrcMgr.getFileEntryForID(SrcMgr.getFileID(SrcLoc));
  return Entry->getName().str();
}

std::string get_source_text_raw(SourceRange range, const SourceManager& SrcMgr)
{
  return Lexer::getSourceText(CharSourceRange::getCharRange(range), SrcMgr, LangOptions()).str();
}

std::string get_source_text(SourceRange range, const SourceManager& SrcMgr)
{
  LangOptions lo;

  // NOTE: SrcMgr.getSpellingLoc() used in case the range corresponds to a macro/preprocessed source.
  auto start_loc = SrcMgr.getSpellingLoc(range.getBegin());
  auto last_token_loc = SrcMgr.getSpellingLoc(range.getEnd());
  auto end_loc = Lexer::getLocForEndOfToken(last_token_loc, 0, SrcMgr, lo);
  auto printable_range = SourceRange{start_loc, end_loc};
  return get_source_text_raw(printable_range, SrcMgr);
}

//-----------------------------------------------------------------------------
// InjectFuzzer - implementation
//-----------------------------------------------------------------------------
void ComputeDeclMatcher::run(const MatchFinder::MatchResult &Result) {

  char FilledBody[0x1000];
  char NewFname[0x100];

  const char *FuzzBodyTemplate = R""""({

    if (!tffuzzing::already_fuzzing && !tffuzzing::was_fuzzed("%1$s")) {

        tffuzzing::already_fuzzing = true;

        tffuzzing::Fuzzer fuzzer = tffuzzing::Fuzzer("%1$s", %2$s);
        OpKernelContext *fuzz_ctx;

        while (fuzzer.has_more_mutations(true)) {
          fuzz_ctx = fuzzer.get_fuzzed_context();
          fuzzer.mut_start_time();
          do_%1$s(fuzz_ctx);
          fuzzer.mut_end_time(fuzz_ctx);
        }

        tffuzzing::already_fuzzing = false;
        do_%1$s(%2$s);
      } else {
        do_%1$s(%2$s);
      }

  })"""";

  std::ifstream in(ONE_TO_ONE_FILE);
  std::string kname;
  std::vector<std::string> knames;

  while (std::getline(in, kname)) {
    if(kname.size() > 0) {
      knames.push_back(kname);
      /* llvm::outs() << kname << " " << kname.length() << "\n"; */
    }
  }

  ASTContext *Ctx = Result.Context;
  const SourceManager &SrcMgr = InjectFuzzerRewriter.getSourceMgr();

  const CXXMethodDecl *ComputeDecl =
    Result.Nodes.getNodeAs<CXXMethodDecl>("computedecl");

  std::string SourceFile = get_source_filename(SrcMgr, ComputeDecl->getLocation());

  if (SourceFile != InputFilename) {
    /* llvm::outs() << SourceFile << " " << InputFilename << "\n"; */
    return;
  }

  if (!ComputeDecl) {
    return;
  }

  bool IsDef = ComputeDecl->isThisDeclarationADefinition();
  const CXXRecordDecl* ParentClass = ComputeDecl->getParent();

  if (!ParentClass) {
    llvm::outs() << "INFO: No parent: " << SourceFile << " " << ComputeDecl->getName() << "\n";
    return;
  }

  StringRef OpName = ParentClass->getName();

  /* Make sure it extends OpKernel */
  const CXXRecordDecl *BaseClass = ParentClass, *TmpBaseClass;
  const TemplateSpecializationType *BaseSpecTemplSpec;
  const TemplateDecl *TemplDecl;

  /* Iterate base classes until there are no more parent classes */
  while (BaseClass->getNumBases() > 0) {

    for (auto &BaseSpecifier : BaseClass->bases()) {

      TmpBaseClass = BaseSpecifier.getType()->getAsCXXRecordDecl();
      if (TmpBaseClass) {
        BaseClass = TmpBaseClass;
        break;
      }

      BaseSpecTemplSpec = BaseSpecifier.getType()->getAs<TemplateSpecializationType>();

      if (BaseSpecTemplSpec) {
        TemplDecl = BaseSpecTemplSpec->getTemplateName().getAsTemplateDecl();
        BaseClass = dyn_cast<CXXRecordDecl>(TemplDecl->getTemplatedDecl());
        break;
      }
    }
  }

  llvm::outs() << "INFO: Base class of " << OpName << ": " << BaseClass->getName() << "\n";
  if (BaseClass->getName().find("OpKernel") == std::string::npos) {
    llvm::outs() << "INFO: " << OpName << " is not OpKernel\n";
    return;
  }

  const CXXRecordDecl* ParentClassDecl = nullptr;
  const ClassTemplateDecl* ParentClassTempl = nullptr;
  const auto Parents = Ctx->getParents(*ComputeDecl);
  const TemplateParameterList *TemplateParams;
  bool HasDeviceTemplate = false;

  /* Find classes with <typename Device> template (have CPU and GPU
   * implementation) */
  for (auto ParentNode : Parents) {
    ParentClassDecl = ParentNode.get<CXXRecordDecl>();
    if (ParentClassDecl)
      ParentClassTempl = ParentClassDecl->getDescribedClassTemplate();
    if (ParentClassTempl) {
      TemplateParams = ParentClassTempl->getTemplateParameters();
      for (auto &TemplParam : *TemplateParams) {
        if (TemplParam->getName().equals("Device")) {
          HasDeviceTemplate = true;
          break;
        }
      }
    }
  }

  StringRef OpNameCheck = StringRef(OpName);
  std::string ToCheck = OpNameCheck.str();

  if (OpNameCheck.endswith("BaseOp") || OpNameCheck.endswith("OpBase")) {
    OpNameCheck = OpNameCheck.drop_back(6);
    ToCheck = OpNameCheck.str() + "Op";
    /* llvm::outs() << OpNameCheck.str() << " " << OpNameCheck.str().length() << "\n"; */
  }

  /* if (std::find(knames.begin(), knames.end(), ToCheck) == knames.end()) { */
  /*   return; */
  /* } else { */
  /*   llvm::outs() << "INFO: Found matching mapped kernel: " << OpNameCheck << "\n"; */
  /* } */

  if (OpName == "OpKernel" || OpName == "AsyncOpKernel"
      || OpName == "UnaryElementWiseOp" || OpName == "HybridAsyncOpKernel") {
    return;
  }

  if (HasDeviceTemplate) {
    llvm::outs() << "INFO: Kernel with Device template parameter: " << OpName << "\n";
  }

  llvm::outs() << "INFO: Found Compute() call in OpKernel child class " << OpName << " (File " << InputFilename << ")\n";

  if (ComputeDecl->getNumParams() > 1) {
    llvm::outs() << "INFO: Skipping " << OpName << " (>1 params)\n";
    return;
  }

  if (ComputeDecl->getNumParams() == 0) {
    llvm::outs() << "INFO: Skipping " << OpName << " (no params)\n";
    return;
  }

  if (ComputeDecl->getStorageClass() == SC_Static) {
    llvm::outs() << "INFO: Skipping " << OpName << " (static)\n";
    return;
  }

  StringRef CtxParamName = ComputeDecl->parameters()[0]->getName();
  FullSourceLoc ComputeStartLoc = Ctx->getFullLoc(ComputeDecl->getBeginLoc());

  std::string ComputeDeclText = get_source_text(SourceRange(ComputeDecl->getBeginLoc(), ComputeDecl->getEndLoc()), SrcMgr);
  std::string DoCallPrefix = "";

  if (ComputeDeclText.find("::Compute(") != std::string::npos) {

    if (ComputeDeclText.find(OpName.str() + "::Compute(") == std::string::npos) {
      bool InParent = false;
      const auto Parents = Ctx->getParents(*ComputeDecl);
      for (auto ParentNode : Parents) {
        if (isa<CXXRecordDecl>(ParentNode.get<Decl>())){
          InParent = true;
          break;
        }
      }
      if (!InParent) {
        llvm::outs() << "INFO: Skipping " << OpName << " (template)\n";
        return;
      }
    } else {
      DoCallPrefix = OpName.str() + "::";
    }
  }

  memset(NewFname, 0, 0x100);
  sprintf(NewFname, "void %sdo_%s(OpKernelContext *%s)", DoCallPrefix.c_str(), OpName.str().c_str(), CtxParamName.str().c_str());

  if (!IsDef) {
    /* Insert the declartion for the wrapped function */
    llvm::outs() << "INFO: Declaration only " << OpName << "File " << InputFilename << "\n";
    InjectFuzzerRewriter.InsertTextAfter(ComputeStartLoc, (Twine(NewFname) + ";\n\t").str());
    return;
  }

  Stmt *ComputeBody = ComputeDecl->getBody();

  if (!ComputeBody && IsDef) {
    llvm::outs() << "INFO: Skipping " << OpName << " (no body)\n";
    return;
  }

  if (CtxParamName.empty()) {
    llvm::outs() << "INFO: Skipping " << OpName << " (no param name)\n";
    return;
  }

  FullSourceLoc ComputeBodyStartLoc = Ctx->getFullLoc(ComputeBody->getBeginLoc());
  SourceRange ComputeSR = ComputeBody->getSourceRange();

  std::string ComputeText = get_source_text(ComputeSR, SrcMgr);

  if (ComputeText.find(std::string("ResourceMgr")) != std::string::npos) {
    llvm::outs() << "INFO: Skipping " << OpName << " (ResourceMgr)\n";
    return;
  }

  if (ComputeText.find(std::string("ResourceHandle")) != std::string::npos) {
    llvm::outs() << "INFO: Skipping " << OpName << " (ResourceHandle)\n";
    return;
  }

  if (ComputeText.find(std::string("mutex")) != std::string::npos ||
      ComputeText.find(std::string("Mutex")) != std::string::npos
      ) {
    llvm::outs() << "INFO: Skipping " << OpName << " (mutex)\n";
    return;
  }

  if (OpName.contains("SummaryOp")) {
    llvm::outs() << "INFO: Skipping " << OpName << " (SummaryOp)\n";
    return;
  }

  if (OpName.contains("AdjustHueOpBase")) {
    llvm::outs() << "INFO: Skipping " << OpName << " (AdjustHueOp)\n";
    return;
  }

  if (SourceFile == KERNEL_DIR + "batch_kernels.cc") {
    llvm::outs() << "INFO: Skipping " << OpName << " (batch kernel)\n";
    return;
  }

  if (SourceFile == KERNEL_DIR + "isotonic_regression_op.cc") {
    llvm::outs() << "INFO: Skipping " << OpName << " (isotonic regression)\n";
    return;
  }

  if (SourceFile == KERNEL_DIR + "fact_op.cc") {
    llvm::outs() << "INFO: Skipping " << OpName << " (fact op)\n";
    return;
  }

  if (SourceFile == KERNEL_DIR + "random_op.cc") {
    llvm::outs() << "INFO: Skipping " << OpName << " (random op)\n";
    return;
  }

  if (SourceFile == KERNEL_DIR + "resource_variable_ops.cc") {
    llvm::outs() << "INFO: Skipping " << OpName << " (resource variable)\n";
    return;
  }

  if (SourceFile == KERNEL_DIR + "list_kernels.cc") {
    llvm::outs() << "INFO: Skipping " << OpName << " (list kernel)\n";
    return;
  }

  if (SourceFile == KERNEL_DIR + "list_kernels.h") {
    llvm::outs() << "INFO: Skipping " << OpName << " (list kernel)\n";
    return;
  }

  if (SourceFile == KERNEL_DIR + "tensor_array_ops.cc") {
    llvm::outs() << "INFO: Skipping " << OpName << " (tensor array)\n";
    return;
  }

  memset(FilledBody, 0, 0x1000);
  sprintf(FilledBody, FuzzBodyTemplate, OpName.str().c_str(), CtxParamName.str().c_str());
  std::string FilledBodyStr(FilledBody);

  InjectFuzzerRewriter.InsertText(ComputeStartLoc, (Twine(NewFname) + ComputeText + "\n\n").str());
  InjectFuzzerRewriter.RemoveText(ComputeSR);
  InjectFuzzerRewriter.InsertText(ComputeBodyStartLoc, FilledBodyStr);

  llvm::outs() << "INFO: Successfully modified " << OpName << "\n";
  return;

}

void ComputeDeclMatcher::onEndOfTranslationUnit() {
  // Replace in place
  InjectFuzzerRewriter.overwriteChangedFiles();

  // Output to stdout
  /* InjectFuzzerRewriter.getEditBuffer(InjectFuzzerRewriter.getSourceMgr().getMainFileID()) */
  /*     .write(llvm::outs()); */
}

InjectFuzzerASTConsumer::InjectFuzzerASTConsumer(Rewriter &R, std::string &InpF) : ComputeDeclHandler(R, InpF) {

  DeclarationMatcher ComputeDeclMatcher =
    cxxMethodDecl(hasName("Compute"))
    .bind("computedecl");

  // InjectFuzzer is the callback that will run when the ASTMatcher finds the pattern
  // above.
  Finder.addMatcher(ComputeDeclMatcher, &ComputeDeclHandler);
}

//-----------------------------------------------------------------------------
// FrotendAction
//-----------------------------------------------------------------------------
class InjectFuzzerPluginAction : public PluginASTAction {
  public:

    bool ParseArgs(const CompilerInstance &,
                   const std::vector<std::string> &Args) override {
      return true;
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef file) override {
      std::string InpF = file.str();
      RewriterForInjectFuzzer.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
      return std::make_unique<InjectFuzzerASTConsumer>(RewriterForInjectFuzzer, InpF);
    }

  private:
    Rewriter RewriterForInjectFuzzer;
};

//-----------------------------------------------------------------------------
// Registration
//-----------------------------------------------------------------------------
static FrontendPluginRegistry::Add<InjectFuzzerPluginAction>
X(/*Name=*/"InjectFuzzer",
  /*Desc=*/"Inject fuzzing code in tensorflow kernels");
