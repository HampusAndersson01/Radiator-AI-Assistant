#!/bin/bash

# Build script for radiator assistant Docker images
# Run this on your Ubuntu server before deploying to Portainer

echo "Building radiator assistant Docker images..."

# Navigate to project directory
cd "$(dirname "$0")"

# Build AI service image
echo "Building AI service..."
docker build -t radiator-ai-service:latest ./ai_service

# Build bot image
echo "Building bot..."
docker build -t radiator-bot:latest ./bot

echo ""
echo "✓ Images built successfully!"
echo ""
echo "You can now deploy the stack in Portainer using docker-compose.portainer.yml"
echo ""
echo "Next steps:"
echo "1. Create a .env file with your credentials:"
echo "   BOT_TOKEN=your_token_here"
echo "   TELEGRAM_WEBHOOK=https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=6483085285"
echo ""
echo "2. In Portainer: Stacks → Add stack → Paste docker-compose.portainer.yml content"
echo "3. Add your environment variables in Portainer"
echo "4. Deploy!"
