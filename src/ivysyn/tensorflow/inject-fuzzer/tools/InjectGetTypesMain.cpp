#include "InjectGetTypes.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// Command line options
//===----------------------------------------------------------------------===//
static llvm::cl::OptionCategory InjectGetTypesCategory("inject-fuzzer options");

//===----------------------------------------------------------------------===//
// PluginASTAction
//===----------------------------------------------------------------------===//
//
std::string InputFilename;

class InjectGetTypesPluginAction : public PluginASTAction {
public:
  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    InjectGetTypesRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<InjectGetTypesASTConsumer>(InjectGetTypesRewriter, InputFilename);
  }

private:
  Rewriter InjectGetTypesRewriter;
};

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//
int main(int Argc, const char **Argv) {
  clang::tooling::CommonOptionsParser OptionsParser(Argc, Argv, InjectGetTypesCategory);
  clang::tooling::ClangTool Tool(OptionsParser.getCompilations(),
                                 OptionsParser.getSourcePathList());

  InputFilename = std::string(Argv[Argc-1]);
  return Tool.run(
      clang::tooling::newFrontendActionFactory<InjectGetTypesPluginAction>().get());
}
