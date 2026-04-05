#!/bin/bash
mkdir -p ~/.streamlit/
echo "\
[theme]\n\
primaryColor = '#667eea'\n\
backgroundColor = '#0a0a1a'\n\
secondaryBackgroundColor = '#1a1a2e'\n\
textColor = '#ccd6f6'\n\
font = 'sans serif'\n\
\n\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
" > ~/.streamlit/config.toml
