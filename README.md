# AI Engineer Portfolio

A clean, professional portfolio website for AI/ML Engineers, designed for easy deployment on GitHub Pages. Features a multi-page layout, responsive design, and integrated Markdown blogging system.

## Features

- **Multi-Page Layout**: Clean structure with dedicated pages for Home, About Me, Projects, and Blog
- **Responsive Design**: Mobile-first design that looks great on all devices
- **Static Site**: Fast, secure, and easily deployable on GitHub Pages
- **Markdown Blogging**: Write blog posts in Markdown and convert them to HTML automatically
- **GitHub Pages Ready**: Structured with `/docs` folder for seamless deployment

## Project Structure

```
/
├── docs/                   # GitHub Pages deployment folder
│   ├── index.html          # Main portfolio page
│   ├── about.html          # About Me page
│   ├── blogs.html          # Blog listing page
│   ├── projects.html       # Projects showcase
│   ├── blog.html           # Individual blog post template
│   ├── resume.pdf          # Your resume (add your own)
│   └── posts/              # Generated HTML blog posts
│       └── blog1.html
├── content/                # Markdown blog posts
│   └── blog1.md
├── publish.py              # Blog generation script
└── README.md               # This file
```

## Setup and Usage

### Prerequisites

- Python 3.6 or higher
- pip package manager

### Installation

1. **Clone the repository**

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

2. **Install dependencies**

```bash
pip install markdown Pygments
```

### Customization

#### 1. Edit HTML Files

Navigate to the `/docs` directory and customize the following:

- **Personal Information**: Update `index.html`, `about.html`, `projects.html`, and `blogs.html` with your details
- **Profile Image**: Replace the placeholder image link in `index.html` with your photo
- **Social Links**: Update GitHub and LinkedIn profile URLs across all HTML files
- **Resume**: Add your actual `resume.pdf` file to the `/docs` directory

#### 2. Add Blog Posts

- Create `.md` files in the `content` directory
- Each file must include front matter metadata (see `content/blog1.md` for format example)
- Write your content using standard Markdown syntax

### Publishing Blog Posts

After creating or editing Markdown files in the `content` directory, generate the HTML pages:

```bash
python publish.py
```

This script converts your Markdown posts to HTML and saves them in `docs/posts/`. **Run this command every time you add or update a blog post.**

## Deployment to GitHub Pages

### 1. Push Your Changes

```bash
git add .
git commit -m "Update portfolio content"
git push
```

### 2. Configure GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** tab
3. Navigate to **Pages** in the left sidebar
4. Under **Build and deployment**:
   - Set **Source** to "Deploy from a branch"
   - Select **Branch**: `main`
   - Select **Folder**: `/docs`
   - Click **Save**

Your site will be live at `https://<your-username>.github.io/<repository-name>` in a few minutes.

## Workflow

1. **Create/Edit** Markdown files in `content/`
2. **Generate** HTML by running `python publish.py`
3. **Commit** changes to Git
4. **Push** to GitHub
5. **Deploy** automatically via GitHub Pages

## Technologies Used

- HTML5 & CSS3
- Python (Markdown processing)
- Markdown & Pygments (syntax highlighting)
- GitHub Pages (hosting)


## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

