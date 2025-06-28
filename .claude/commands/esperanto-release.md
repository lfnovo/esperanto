
- Bump the version in @pyproject.toml
- Run `uv sync --all-extras` locally
- Commit and push, merge to main.
- Don't forget to commit the uv.lock file too
- Generate a new tag
- Generate a new release via the Github cli. 
(the github action workflow will publish it to pypi)