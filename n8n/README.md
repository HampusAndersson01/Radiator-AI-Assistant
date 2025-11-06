# n8n Workflow Configuration

This directory contains the n8n workflow for the Smart Radiator Assistant.

## Setup Instructions

### 1. Environment Variables

Configure the following environment variables in your n8n instance:

```bash
HOME_ASSISTANT_URL=https://your-home-assistant-instance.com
HOME_ASSISTANT_TOKEN=your_long_lived_access_token
AI_SERVICE_URL=http://your-ai-service-host:8000
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### 2. n8n Credentials

#### Telegram Credentials
1. In n8n, go to **Credentials** → **New**
2. Select **Telegram API**
3. Enter your bot token from @BotFather
4. Save the credentials

### 3. Workflow Configuration

After importing the workflow, update the following:

1. **Telegram Node** - Select your configured Telegram credentials
2. **Webhook ID** - This will be auto-generated when you activate the workflow

### 4. Getting Your Credentials

#### Home Assistant Long-Lived Token
1. In Home Assistant, go to your Profile
2. Scroll to **Long-Lived Access Tokens**
3. Click **Create Token**
4. Give it a name (e.g., "n8n Radiator Workflow")
5. Copy the token and add it to your environment variables

#### Telegram Chat ID
1. Message your bot
2. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Look for the `chat.id` field in the response
4. Use this as your `TELEGRAM_CHAT_ID`

## Workflow Overview

The workflow consists of two main flows:

### Training Flow (Every 15 minutes)
1. Fetches current Home Assistant states
2. Retrieves radiator levels from AI service
3. Builds training payloads
4. Trains the AI model with current data

### Prediction Flow (Every 4 hours)
1. Fetches current Home Assistant states
2. Builds prediction payloads
3. Gets AI recommendations
4. Sends Telegram notification if adjustments are recommended

## Security Notes

- Never commit credentials to git
- Use environment variables for all sensitive data
- Regularly rotate access tokens
- Review access logs periodically
