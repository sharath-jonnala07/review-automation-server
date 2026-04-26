# server/mcp_servers/smtp_server.py
"""
SMTP-based email MCP server.
Sends emails via Gmail SMTP (works with personal Gmail + app password).
Replaces Gmail API's domain-wide delegation approach.
"""
import os
import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import structlog

logger = structlog.get_logger()
app = FastAPI(title="Pulse SMTP MCP Server")

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class MCPToolRequest(BaseModel):
    """MCP HTTP wrapper request."""

    arguments: dict[str, Any] = Field(default_factory=dict)


_drafts: dict[str, dict[str, str]] = {}


def _send_email_message(
    to: str,
    subject: str,
    html_body: str,
    plain_text_body: str,
) -> dict[str, Any]:
    """Send an email immediately via Gmail SMTP."""
    _validate_credentials()
    gmail_sender = os.getenv("GMAIL_SENDER", "")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD", "")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_sender
    msg["To"] = to
    if plain_text_body:
        msg.attach(MIMEText(plain_text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_sender, gmail_app_password)
        server.sendmail(gmail_sender, [to], msg.as_string())

    logger.info("email_sent", to=to, subject=subject)
    return {"success": True, "message": f"Email sent to {to}", "id": subject}


def _validate_credentials() -> None:
    """Validate SMTP credentials are configured."""
    gmail_sender = os.getenv("GMAIL_SENDER", "")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD", "")
    if not gmail_sender:
        raise RuntimeError(
            "GMAIL_SENDER is required. Set it to your Gmail address in .env"
        )
    if not gmail_app_password:
        raise RuntimeError(
            "GMAIL_APP_PASSWORD is required. Generate an app password at "
            "https://myaccount.google.com/apppasswords (requires 2FA enabled)"
        )


@app.post("/send_email")
async def send_email(request: MCPToolRequest) -> dict[str, Any]:
    """
    Send an email via Gmail SMTP.
    
    Arguments:
        to: Recipient email address
        subject: Email subject
        html_body: Email body (HTML)
        plain_text_body: Plain text fallback (optional)
    
    Returns:
        {"success": True, "message": "Email sent successfully"}
    """
    _validate_credentials()
    gmail_sender = os.getenv("GMAIL_SENDER", "")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD", "")
    
    args = request.arguments
    to = args.get("to")
    subject = args.get("subject")
    html_body = args.get("html_body")
    plain_text_body = args.get("plain_text_body", "")
    
    if not all([to, subject, html_body]):
        raise HTTPException(status_code=400, detail="Missing required arguments: to, subject, html_body")
    
    try:
        return _send_email_message(to, subject, html_body, plain_text_body)
    
    except smtplib.SMTPAuthenticationError as e:
        logger.error("smtp_auth_failed", error=str(e))
        raise HTTPException(
            status_code=401,
            detail=f"Gmail auth failed: {str(e)}. Check GMAIL_SENDER and GMAIL_APP_PASSWORD."
        )
    except Exception as e:
        logger.error("email_send_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")


@app.post("/create_draft_email")
async def create_draft_email(request: MCPToolRequest) -> dict[str, Any]:
    """
    Create a draft email (returns the composed message without sending).
    Useful for preview/confirmation.
    
    Arguments:
        to: Recipient email address
        subject: Email subject
        html_body: Email body (HTML)
    
    Returns:
        {"draft": "email content preview"}
    """
    gmail_sender = os.getenv("GMAIL_SENDER", "")
    args = request.arguments
    to = args.get("to")
    subject = args.get("subject")
    html_body = args.get("html_body")
    
    if not all([to, subject, html_body]):
        raise HTTPException(status_code=400, detail="Missing required arguments: to, subject, html_body")
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_sender
    msg["To"] = to
    msg.attach(MIMEText(html_body, "html"))
    
    return {
        "draft": msg.as_string(),
        "preview": f"To: {to}\nSubject: {subject}\n\n[HTML body preview]"
    }


@app.post("/tools/search_messages")
async def search_messages(request: MCPToolRequest) -> dict[str, Any]:
    """SMTP transport cannot search Gmail; return no prior messages."""
    _ = request
    return {"messages": []}


@app.post("/tools/create_draft")
async def create_draft(request: MCPToolRequest) -> dict[str, Any]:
    """Create an in-memory draft compatible with the Gmail MCP client."""
    message = request.arguments.get("message")
    if not isinstance(message, dict):
        raise HTTPException(status_code=400, detail="message is required")

    to = str(message.get("to") or "").strip()
    subject = str(message.get("subject") or "").strip()
    body = message.get("body") or {}
    html_body = str(body.get("html") or "")
    text_body = str(body.get("text") or "")
    if not all([to, subject, html_body]):
        raise HTTPException(status_code=400, detail="message.to, message.subject, and message.body.html are required")

    draft_id = str(len(_drafts) + 1)
    _drafts[draft_id] = {
        "to": to,
        "subject": subject,
        "html_body": html_body,
        "text_body": text_body,
    }
    return {"id": draft_id}


@app.post("/tools/send_message")
async def send_message(request: MCPToolRequest) -> dict[str, Any]:
    """Send an in-memory draft compatible with the Gmail MCP client."""
    draft_id = str(request.arguments.get("draftId") or "").strip()
    draft = _drafts.get(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="draft not found")

    try:
        result = _send_email_message(
            draft["to"],
            draft["subject"],
            draft["html_body"],
            draft["text_body"],
        )
    except smtplib.SMTPAuthenticationError as e:
        logger.error("smtp_auth_failed", error=str(e))
        raise HTTPException(status_code=401, detail=f"Gmail auth failed: {str(e)}")
    except Exception as e:
        logger.error("email_send_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

    _drafts.pop(draft_id, None)
    return {"id": result["id"]}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "SMTP Email MCP Server"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5001)