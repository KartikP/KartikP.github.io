---
title: 'How to embed Jupyter Notebooks (.ipynb) into Jekyll (AcademicPages)'
date: 2023-09-19
permalink: /posts/2023/09/Convert-Juypter-Notebook-to-Jekyll/
tags:
  - data science
  - python
  - code
  - jupyter-notebooks
  - jekyll
---

Since creating this personal website, I've wanted to find a way to embed python notebook snippets to showcase my projects and hopefully eventually create tutorials that people can learn from. To do this, you need to convert python notebook files (.ipynb) into markdown (.md) along with all the embeded images.

1. Create a `_notebooks` directory inside the master folder. This is where you should be putting your jupyter notebooks.
2. Using terminal, navigate to the `_notebooks` directory and type the following command to convert it to markdown `jupyter nbconvert <name of file>.ipynb --to markdown`
3. Move the converted products (which should be the .md file along with a folder containing an image outputs) to `_posts` folder.
4. Add the Jekyll AcademicPages metadata.
5. Add `../` infront of every image path. This will allow the path of the image (which is in the `_posts` folder) to be read. Alternatively, you can place the images in the `_images` folder but will need to adjust the path accordingly.

There you have it! Pretty simple. Lets you turn your website into an all-in-one repository for all your projects.
