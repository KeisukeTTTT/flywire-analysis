# CLAUDE.md

## 実行ルール
- Pythonファイルを実行する際は、必ず `uv run` を用いること。
  - 例: `uv run python script.py`

## コミットルール
- コミットメッセージは Conventional Commits に従うこと。
  - 例: `feat: add data preprocessing pipeline`
  - 例: `fix: handle missing values in loader`
- コミットは適切な単位で分割すること（機能追加・修正・リファクタリングを混在させない）。

## 問題管理ルール
- バグ・設計課題・改善提案のうち、追跡・議論・タスク分解が必要なものは **GitHub Issues** (`gh issue create`) で管理する。
  - `docs/` や `plans/` にIssue原稿を重複して置かない（情報の一元化）。
- 軽微で自己完結する変更（例: typo修正、単純なリネーム、小さなドキュメント修正）は、新規 Issue を立てずにそのままコミットしてよい。
- 実装計画・TODOリストは **Issue のコメント** にタスクリスト (`- [ ]`) として追記する。
  - 進捗はチェックボックスの更新またはコメントで報告する。
- 既存 Issue に関連するコミットだけ、コミットメッセージに `refs #<issue番号>` を含めて Issue と自動リンクさせる。
  - 例: `fix: prevent decay collapse in deep shells (refs #5)`
- Issue と無関係なコミットには、無理に Issue 番号を付けない。
- Issue 完了時は `gh issue close <番号>` でクローズする。
