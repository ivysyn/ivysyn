#include "InjectFuzzer.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// Command line options
//===----------------------------------------------------------------------===//
static llvm::cl::OptionCategory InjectFuzzerCategory("inject-fuzzer options");

//===----------------------------------------------------------------------===//
// PluginASTAction
//===----------------------------------------------------------------------===//
//
std::string InputFilename;

class InjectFuzzerPluginAction : public PluginASTAction {
public:
  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    return true;
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    InjectFuzzerRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<InjectFuzzerASTConsumer>(InjectFuzzerRewriter, InputFilename);
  }

private:
  Rewriter InjectFuzzerRewriter;
};

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//
int main(int Argc, const char **Argv) {
  clang::tooling::CommonOptionsParser OptionsParser(Argc, Argv, InjectFuzzerCategory);
  clang::tooling::ClangTool Tool(OptionsParser.getCompilations(),
                                 OptionsParser.getSourcePathList());

  InputFilename = std::string(Argv[Argc-1]);
  return Tool.run(
      clang::tooling::newFrontendActionFactory<InjectFuzzerPluginAction>().get());
}
