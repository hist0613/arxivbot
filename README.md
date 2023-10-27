# arxivbot
설명

## Usage
1. Install requirements
```sh
pip install -r requirements.txt
```

2. Define `WORKSPACES` in settings.py
```python
WORKSPACES = [
    {
        'workspace': '{workspace}',
        'allowed_channel': '{channel_name}',
        'allowed_channel_id': '{channel_id}',
        'slack_token': "{app_slack_token}",
        'fields': ["{arxiv_field}",]
    },
]
```

Example
```python
WORKSPACES = [
    {
        'workspace': 'ai-research-kr',
        'allowed_channel': 'nlp-research',
        'allowed_channel_id': 'X000XXXXX0X',
        'slack_token': "xxxx-1234567890-1234567890123-X0xx0x0xxXXxxxxXx0x0x0XX",
        'fields': ["cs.CL", "cs.IR"]
    },
]
```

3. Define `OPENAI_API_KEY` in settings.py
```python
OPENAI_API_KEY = "sheep-duck-is-very-good-large-language-model"
```