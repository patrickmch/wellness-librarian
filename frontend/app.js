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
        sessionId: 'session-' + Date.now(), // Session ID for feedback tracking
        currentSources: [], // Sources for current message (used by citation links)
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

            // Build conversation history from previous messages (excluding the current one)
            // Only include role and content fields for the API
            const history = this.messages.map(m => ({
                role: m.role,
                content: m.content,
            }));

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
                        history: history.length > 0 ? history : null,
                        session_id: this.sessionId,
                    }),
                });

                if (!response.ok) {
                    throw new Error(`API error: ${response.status}`);
                }

                const data = await response.json();

                // Add assistant message with query reference for feedback
                this.messages.push({
                    id: this.generateMessageId(),
                    role: 'assistant',
                    content: data.response,
                    sources: data.sources,
                    query: message, // Store original query for feedback
                    feedback: null, // Will be 'up' or 'down' after user rates
                    showExcerpts: false, // For glass box toggle
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

        formatMessage(content, sources = []) {
            // Store sources locally for citation link building
            const msgSources = sources || [];

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

            // Links [text](url) - but NOT citation-style [1], [2] etc
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

            // Convert citation references [1], [2], etc. to clickable links
            html = html.replace(/\[(\d+)\]/g, (match, num) => {
                const idx = parseInt(num) - 1;
                const source = msgSources[idx];
                if (!source) return match;

                // Build deep link inline to avoid 'this' context issues
                const t = source.start_time_seconds || 0;
                let deepLink;
                if (source.source === 'youtube') {
                    deepLink = source.video_url.includes('?')
                        ? `${source.video_url}&t=${t}`
                        : `${source.video_url}?t=${t}`;
                } else {
                    deepLink = `${source.video_url}#t=${t}s`;
                }

                const safeTitle = (source.title || '').replace(/"/g, '&quot;');
                return `<a href="${deepLink}" target="_blank" rel="noopener" class="citation-link" title="${safeTitle}">[${num}]</a>`;
            });

            return html;
        },

        buildDeepLink(source) {
            const t = source.start_time_seconds || 0;
            if (source.source === 'youtube') {
                // YouTube: append &t=123 or ?t=123
                return source.video_url.includes('?')
                    ? `${source.video_url}&t=${t}`
                    : `${source.video_url}?t=${t}`;
            } else {
                // Vimeo: append #t=123s
                return `${source.video_url}#t=${t}s`;
            }
        },

        async submitFeedback(messageId, type) {
            const msg = this.messages.find(m => m.id === messageId);
            if (!msg || msg.feedback === type) return; // Already submitted this type

            try {
                await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message_id: messageId,
                        feedback_type: type,
                        session_id: this.sessionId,
                        query: msg.query,
                    }),
                });
                msg.feedback = type;
            } catch (error) {
                console.error('Failed to submit feedback:', error);
            }
        },

        toggleExcerpts(msg) {
            msg.showExcerpts = !msg.showExcerpts;
        },
    };
}
