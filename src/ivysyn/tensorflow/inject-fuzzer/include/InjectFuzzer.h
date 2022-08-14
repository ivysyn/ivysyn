#ifndef INJECT_FUZZER_H
#define INJECT_FUZZER_H

#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Rewrite/Core/Rewriter.h"

class ComputeDeclMatcher : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  ComputeDeclMatcher(clang::Rewriter &InjectFuzzerRewriter, std::string InputFilename) :
    InjectFuzzerRewriter(InjectFuzzerRewriter), InputFilename(InputFilename) {}
  // Callback that's executed whenever the Matcher in InjectFuzzerASTConsumer
  // matches.
  void run(const clang::ast_matchers::MatchFinder::MatchResult &) override;
  // Callback that's executed at the end of the translation unit
  void onEndOfTranslationUnit() override;

private:
  clang::Rewriter InjectFuzzerRewriter;
  std::string InputFilename;
};

class InjectFuzzerASTConsumer : public clang::ASTConsumer {
public:

  InjectFuzzerASTConsumer(clang::Rewriter &R, std::string &InputFilename);

  void HandleTranslationUnit(clang::ASTContext &Ctx) override {
    Finder.matchAST(Ctx);
  }

  clang::StringRef GetOpName(clang::ASTContext *Ctx, const clang::FunctionDecl *FDecl);

private:
  clang::ast_matchers::MatchFinder Finder;
  ComputeDeclMatcher ComputeDeclHandler;
};

#endif
