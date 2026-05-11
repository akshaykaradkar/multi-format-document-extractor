Here is a complete technical documentation of the "Smart Chapter Summarizer" feature we built, formatted in Markdown. You can use this directly for your internal engineering wiki, pull request description, or manager review.

# ---

**Technical Documentation: Smart Chapter Summarization Sub-Agent**

## **1\. Overview**

The **Smart Chapter Summarization Sub-Agent** is a new capability added to the YouTube Analyst (YouBuddy) ADK orchestrator. It allows the system to autonomously ingest a YouTube video by its ID, extract its core narrative, and generate a high-density "Executive Summary."

Rather than relying purely on the creator's metadata, this agent reads the actual spoken transcript to generate "Smart Chapters" (timestamped topic shifts) and cross-references them with top community comments to gauge true audience sentiment.

## **2\. Architectural Approach**

The feature was built using a **"Power Tool" pattern**. Instead of forcing the LLM to make 3 or 4 separate reasoning loops to gather metadata, transcripts, and comments individually (which increases latency and token costs), we engineered a single backend Python function that bundles all necessary context in one fast, concurrent pass.

### **2.1 The "Power Tool" (fetch\_video\_context\_for\_summary)**

Located in youtube\_analyst/tools.py, this function handles all data ingestion.

* **Metadata Extraction:** Uses the official YouTube Data API v3 (youtube.videos().list) to fetch the video's Title and Description. *(Cost: 1 Quota Unit)*  
* **Transcript Extraction:** Uses the youtube-transcript-api library to scrape the hidden caption data directly from the YouTube player. It is configured with language fallbacks (\['en', 'en-US', 'en-GB'\]) and will default to auto-generated captions if manual ones do not exist. *(Cost: 0 Quota Units)*  
* **Community Extraction:** Uses the YouTube Data API v3 (youtube.commentThreads().list) to fetch the top 15 comments sorted by relevance. *(Cost: 1 Quota Unit)*

**Total API Cost:** 2 Quota Units per summary. Highly optimized.

### **2.2 Fallback & Resilience Logic**

Because the youtube-transcript-api acts as a web scraper, it is susceptible to being blocked by YouTube's anti-bot protections (especially when running from corporate data center IPs like Corp Airlock).

* **The Catch:** The tool wraps the transcript fetch in a try/except block.  
* **The Pivot:** If an Exception is caught (e.g., HTTP 403 or Captions Disabled), it returns a "Transcript unavailable" flag to the LLM.  
* **The Resolution:** The LLM's system prompt is explicitly programmed to pivot to using the **Video Description** to generate the summary if the transcript is missing, ensuring the agent never crashes and always returns a payload to the user.

## **3\. Agent Definition & Prompting**

### **3.1 Sub-Agent Instantiation (summarization\_agent.py)**

The sub-agent is defined using the Google Agent Development Kit (google.adk.agents.Agent). It is isolated from the root agent to ensure its context window remains focused solely on summarization tasks.

* **Model:** Inherits the global Gemini model (e.g., gemini-2.5-flash) via GeminiWithLocation.  
* **Tools Bindings:** Only bound to the fetch\_video\_context\_for\_summary tool to prevent tool-hallucination.

### **3.2 System Prompt (prompts/summarization\_agent.txt)**

The prompt enforces a strict UI/UX format for the output:

1. **TL;DR:** A 2-sentence overarching summary.  
2. **Smart Chapters:** The LLM scans the transcript text (which has \[MM:SS\] injected into the strings) and autonomously identifies 3 to 5 narrative shifts, formatting them as timestamped bullet points.  
3. **Community Verdict:** The LLM analyzes the array of top comments and writes a brief consensus report on whether the audience agreed with the creator's points.

## **4\. Integration with the Root Orchestrator**

To make this capability available to the user, the sub-agent was registered with the main youtube\_agent orchestrator in youtube\_analyst/agent.py.

1. **Delegation:** summarization\_agent was appended to the sub\_agents=\[\] list of the root youtube\_agent. The ADK framework automatically wraps sub-agents as callable tools for the root LLM.  
2. **Routing Instructions:** The root agent's system prompt (prompts/youtube\_agent.txt) was updated in the Workflow: section with the following directive:  
   * *"If the user explicitly asks to summarize a video or asks for a TL;DR for a specific video ID, IMMEDIATELY delegate the task to the summarization\_agent."*

## **5\. Libraries & Dependencies Used**

No new dependencies were added to the pyproject.toml. We utilized the existing approved stack:

* google-adk: For Agent, ToolContext, and Runner orchestration.  
* google-api-python-client: For authorized requests to the YouTube Data API v3.  
* youtube-transcript-api: For extracting raw SRT/XML caption data.  
* google-genai: The underlying SDK for Gemini model interactions.