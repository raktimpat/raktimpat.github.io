import os
import markdown
import frontmatter

def publish_blog_posts():
    """
    Reads Markdown files from the 'content' directory, converts them to HTML,
    and injects them into the blog template.
    """
    content_dir = 'content'
    output_dir = 'docs'
    post_dir = os.path.join(output_dir, 'posts')
    template_path = os.path.join(output_dir, 'blog.html')

    # Ensure the output directory for blogs exists
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found.")
        return

    # Read the main blog page template
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_html = f.read()
    except FileNotFoundError:
        print(f"Error: Blog template '{template_path}' not found.")
        return

    print("Starting blog publication process...")

    # Process each markdown file in the content directory
    for filename in os.listdir(content_dir):
        if filename.endswith('.md'):
            md_path = os.path.join(content_dir, filename)
            
            try:
                # Parse the markdown file with frontmatter
                post = frontmatter.load(md_path)
                metadata = post.metadata
                content_md = post.content

                # Check for required metadata
                if 'title' not in metadata or 'date' not in metadata or 'author' not in metadata:
                    print(f"Warning: Skipping '{filename}' due to missing metadata (title, date, or author).")
                    continue

                # Convert markdown content to HTML
                html_content = markdown.markdown(content_md, extensions=['fenced_code', 'codehilite'])

                # Replace placeholders in the template
                post_html = template_html.replace('{{POST_TITLE}}', metadata.get('title', 'Untitled Post'))
                post_html = post_html.replace('{{POST_AUTHOR}}', metadata.get('author', 'Anonymous'))
                post_html = post_html.replace('{{POST_DATE}}', metadata.get('date', ''))
                post_html = post_html.replace('{{POST_CONTENT}}', html_content)
                
                # Create the output HTML filename
                html_filename = os.path.splitext(filename)[0] + '.html'
                output_path = os.path.join(post_dir, html_filename)

                # Write the final HTML file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(post_html)
                
                print(f"Successfully published '{filename}' to '{html_filename}'")

            except Exception as e:
                print(f"Error processing '{filename}': {e}")

    print("\nBlog publication complete.")
    print("Please check the 'docs/posts' folder for the generated HTML files.")


if __name__ == '__main__':
    # --- Instructions ---
    # 1. Make sure you have the required libraries installed:
    #    pip install markdown python-frontmatter Pygments
    #
    # 2. Place your blog posts as .md files in the 'content' directory.
    #
    # 3. Each .md file MUST have YAML front matter at the top, like this:
    #    ---
    #    title: "Your Awesome Blog Post Title"
    #    author: "Your Name"
    #    date: "October 17, 2025"
    #    summary: "A short summary of your post."
    #    ---
    #    (Your markdown content starts here...)
    #
    # 4. Run this script from the root directory of your project:
    #    python publish.py
    # --------------------
    publish_blog_posts()

