## Build the Docs

### Setup

Install Doxygen:

```
brew install doxygen
```

Install Python packages:

```
pip install -r requirements.txt
```

### Build

Build the docs from `mlx-c/docs/`

```
doxygen && make html
```

View the docs by running a server in `mlx-c/docs/`:

```
python -m http.server <port>
```

and point your browser to `http://localhost:<port>`.

### Push to GitHub Pages

Check-out the `gh-pages` branch (`git switch gh-pages`) and build
the docs. Then force add the `build/html` directory:

`git add -f build/html`

Commit and push the changes to the `gh-pages` branch.
