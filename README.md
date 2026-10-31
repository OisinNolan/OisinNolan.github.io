# oisinnolan.github.io

Personal site built with Jekyll and served by GitHub Pages.

- `index.html` is the about page and `posts.html` lists the posts.
- Posts go in `_posts/`. Front matter keys: `title`, `description` (shown in the list), `tags`, and `math: true` to load KaTeX.
- Large standalone pages (for example the ultrasound viewers) are hosted in the Hugging Face Space `oisinn/website_files` (see `hf_upload.py`). Hugging Face serves them as downloads, so `viewer/index.html` fetches one and displays it: `/viewer/?f=<name>` loads `<name>.html` from the Space. To list one, add a stub post with `layout: redirect` and `link: /viewer/?f=<name>`.
- Profile links are set in `_config.yml` under `links`.
