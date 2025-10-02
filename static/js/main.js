document.addEventListener('DOMContentLoaded', () => {
    // Find all code blocks
    const codeBlocks = document.querySelectorAll('.codehilite');

    codeBlocks.forEach(block => {
        // Create a "Copy" button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-btn';
        copyButton.innerHTML = '<i class="far fa-copy"></i> Copy';

        // Create a wrapper div and add the block and button to it
        const wrapper = document.createElement('div');
        wrapper.className = 'code-wrapper';
        
        // Place the wrapper next to the code block, then move the block inside it
        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(block);
        wrapper.appendChild(copyButton);

        // Add click event listener to the button
        copyButton.addEventListener('click', () => {
            const code = block.querySelector('pre').innerText;
            navigator.clipboard.writeText(code).then(() => {
                // Give user feedback
                copyButton.innerHTML = '<i class="fas fa-check"></i> Copied!';
                setTimeout(() => {
                    copyButton.innerHTML = '<i class="far fa-copy"></i> Copy';
                }, 2000); // Reset after 2 seconds
            });
        });
    });
});