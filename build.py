import os
import markdown
from jinja2 import Environment, FileSystemLoader
import yaml

# --- Setup ---
output_dir = "blog"
posts_dir = "_blogs"
templates_dir = "_templates"

env = Environment(loader=FileSystemLoader(templates_dir))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Collect and Parse Blog Posts ---
posts = []
for filename in os.listdir(posts_dir):
    if filename.endswith(".md"):
        filepath = os.path.join(posts_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                md_content = parts[2]
            else:
                frontmatter, md_content = {}, content

            html_content = markdown.markdown(md_content, extensions=['fenced_code', 'codehilite'])
            
            posts.append({
                'title': frontmatter.get('title', 'Untitled Post'),
                'date': frontmatter.get('date', ''),
                'summary': frontmatter.get('summary', ''),
                'content': html_content,
                'slug': os.path.splitext(filename)[0]
            })

# --- Generate Individual Blog Pages ---
post_template = env.get_template("post.html")
for post in posts:
    output_path = os.path.join(output_dir, f"{post['slug']}.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(post_template.render(post=post, title=post['title'], date=post['date'], content=post['content']))
    print(f"Generated post: {output_path}")

# --- Generate Main Pages (index, all_blogs, projects) ---
# For each page, we get the template and render it.
pages_to_build = {
    "base.html": "index.html",
    "blog_all.html": "blog.html",
    "projects_all.html": "projects.html" # We need to create this template
}

for template_name, output_name in pages_to_build.items():
    template = env.get_template(template_name)
    with open(output_name, 'w', encoding='utf-8') as f:
        # Pass the 'posts' variable to any template that might need it
        f.write(template.render(posts=posts))
    print(f"Generated page: {output_name}")

print("\nBuild complete! ✨")