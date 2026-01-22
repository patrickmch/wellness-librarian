/**
 * Wellness Librarian - Chat Application
 * Alpine.js application logic
 */

function chatApp() {
    return {
        messages: [],
        inputMessage: '',
        isLoading: false,
        messageIdCounter: 0, // Unique ID counter for messages
        stats: {
            total_videos: 0,
            total_chunks: 0,
        },

        async init() {
            // Load stats on init
            await this.loadStats();
        },

        // Generate unique message ID
        generateMessageId() {
            return `msg-${Date.now()}-${++this.messageIdCounter}`;
        },

        async loadStats() {
            try {
                const response = await fetch('/api/sources');
                if (response.ok) {
                    const data = await response.json();
                    this.stats = {
                        total_videos: data.total_videos,
                        total_chunks: data.total_chunks,
                    };
                }
            } catch (error) {
                console.error('Failed to load stats:', error);
            }
        },

        async sendMessage() {
            const message = this.inputMessage.trim();
            if (!message || this.isLoading) return;

            // Add user message
            this.messages.push({
                id: this.generateMessageId(),
                role: 'user',
                content: message,
            });

            this.inputMessage = '';
            this.isLoading = true;
            this.scrollToBottom();

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        stream: false,
                    }),
                });

                if (!response.ok) {
                    throw new Error(`API error: ${response.status}`);
                }

                const data = await response.json();

                // Add assistant message
                this.messages.push({
                    id: this.generateMessageId(),
                    role: 'assistant',
                    content: data.response,
                    sources: data.sources,
                });

            } catch (error) {
                console.error('Chat error:', error);

                // Add error message
                this.messages.push({
                    id: this.generateMessageId(),
                    role: 'assistant',
                    content: "I apologize, but I'm having trouble connecting to the library right now. Please try again in a moment.",
                    sources: [],
                });
            } finally {
                this.isLoading = false;
                this.scrollToBottom();
            }
        },

        askQuestion(question) {
            this.inputMessage = question;
            this.sendMessage();
        },

        scrollToBottom() {
            this.$nextTick(() => {
                const container = this.$refs.messagesContainer;
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            });
        },

        formatMessage(content) {
            // Convert markdown-like formatting to HTML
            let html = content;

            // Escape HTML
            html = html
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');

            // Bold **text**
            html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

            // Italic *text*
            html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

            // Links [text](url)
            html = html.replace(
                /\[([^\]]+)\]\(([^)]+)\)/g,
                '<a href="$2" target="_blank" rel="noopener">$1</a>'
            );

            // Headers (### Header)
            html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
            html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
            html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

            // Bullet lists
            html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
            html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

            // Numbered lists
            html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

            // Blockquotes
            html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

            // Code blocks ```code```
            html = html.replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>');

            // Inline code `code`
            html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

            // Paragraphs (double newlines)
            html = html.replace(/\n\n/g, '</p><p>');
            html = '<p>' + html + '</p>';

            // Clean up empty paragraphs
            html = html.replace(/<p>\s*<\/p>/g, '');

            // Single newlines to <br>
            html = html.replace(/\n/g, '<br>');

            return html;
        },
    };
}
