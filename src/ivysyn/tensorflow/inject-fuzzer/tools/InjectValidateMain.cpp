#include "InjectValidate.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// Command line options
//===----------------------------------------------------------------------===//
static llvm::cl::OptionCategory InjectValidateCategory("inject-fuzzer options");

//===----------------------------------------------------------------------===//
// PluginASTAction
//===----------------------------------------------------------------------===//
//
std::string InputFilename;

class InjectValidatePluginAction : public PluginASTAction {
public:
  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    InjectValidateRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<InjectValidateASTConsumer>(InjectValidateRewriter, InputFilename);
  }

private:
  Rewriter InjectValidateRewriter;
};

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//
int main(int Argc, const char **Argv) {
  clang::tooling::CommonOptionsParser OptionsParser(Argc, Argv, InjectValidateCategory);
  clang::tooling::ClangTool Tool(OptionsParser.getCompilations(),
                                 OptionsParser.getSourcePathList());

  InputFilename = std::string(Argv[Argc-1]);
  return Tool.run(
      clang::tooling::newFrontendActionFactory<InjectValidatePluginAction>().get());
}
