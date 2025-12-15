# Business Case & Risk Assessment

## Document Automation & Data Harmonization System

**Author:** Akshay Karadkar
**Date:** December 2025

---

## 1. Executive Summary

This document automation system addresses the challenge of processing purchase orders from multiple clients with varying document formats (PDF, Excel, Word, CSV, scanned images) and transforming them into standardized JSON output.

**Key Numbers:**
- Implementation: ₹12-15 Lakhs (one-time)
- Annual Operating: ₹5-7 Lakhs
- Annual Savings: ₹10-15 Lakhs
- Payback: ~12-14 months

---

## 2. Current State Analysis

### Cost Breakdown (800 orders/month scenario)

| Category | Monthly | Annual |
|----------|---------|--------|
| Data Entry Staff (3 FTE @ ₹25K) | ₹75,000 | ₹9,00,000 |
| Supervisor time (0.5 FTE @ ₹50K) | ₹25,000 | ₹3,00,000 |
| Error correction & rework | ₹30,000 | ₹3,60,000 |
| **Total Current Cost** | **₹1,30,000** | **₹15,60,000** |

**Salary References:**
- Data Entry Operator: ₹20-30K/month — Glassdoor India, Indeed India (Dec 2024)
- Senior Executive: ₹45-60K/month — Naukri.com salary insights

### Current Process Issues

| Issue | Impact |
|-------|--------|
| Processing time | 2-4 hours per order |
| Error rate | 5-8% (industry typical for manual entry) |
| Scalability | Need additional staff for growth |
| Consistency | Format variations cause integration issues |

---

## 3. Proposed Solution

### Hybrid AI Approach

The system uses a smart routing strategy:

```
Rule-based parsing (free) → For structured docs (PDF, Excel, CSV)
AI extraction (GPT-4o)   → Only for complex/scanned documents
```

**Why this matters for cost:**
- ~70% of documents can be processed with rule-based parsing (zero API cost)
- Only 30% require paid AI extraction
- Results in 60-70% lower costs vs. using AI for everything

### Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Processing time | 2-4 hours | <1 minute | 98%+ |
| Error rate | 5-8% | <1% | 85%+ |
| Staff needed | 3 FTE | 0.5 FTE (review only) | 83% |
| Scalability | Linear cost | Near-zero marginal cost | 10x+ |

---

## 4. Cost Analysis

### Implementation Costs (One-time)

| Item | Estimate |
|------|----------|
| Development (3-4 months, 2 devs) | ₹8-10 Lakhs |
| Cloud setup & integration | ₹2-3 Lakhs |
| Testing & training | ₹1-2 Lakhs |
| **Total** | **₹12-15 Lakhs** |

**Developer rates (India, Dec 2024):**
- Senior Python Developer: ₹80K-1.5L/month — LinkedIn Salary
- AI/ML Engineer: ₹1-2L/month — Naukri.com

### Operating Costs (Annual)

| Item | Monthly | Annual |
|------|---------|--------|
| Cloud (AWS/Azure) | ₹15-25K | ₹2-3 Lakhs |
| OpenAI API (hybrid mode) | ₹15-20K | ₹2-2.5 Lakhs |
| Maintenance | ₹10K | ₹1.2 Lakhs |
| **Total** | **₹40-55K** | **₹5-7 Lakhs** |

**API Cost Calculation (OpenAI GPT-4o):**
- Pricing: $2.50/1M input tokens, $10/1M output tokens (platform.openai.com/pricing)
- Per document (AI mode): ~₹15-25
- Per document (hybrid mode, 70% rule-based): ~₹5-8 average
- 800 docs × ₹8 = ₹6,400/month for extraction

---

## 5. ROI Analysis

### Annual Savings

| Category | Savings |
|----------|---------|
| Labor reduction (2.5 FTE) | ₹7-9 Lakhs |
| Error reduction (85%) | ₹3-4 Lakhs |
| Productivity & opportunity | ₹2-3 Lakhs |
| **Total Annual Savings** | **₹12-16 Lakhs** |

### Payback Calculation

```
Implementation Cost:      ₹14 Lakhs (mid-estimate)
Annual Operating Cost:    ₹6 Lakhs
Annual Savings:           ₹14 Lakhs

Net Annual Benefit:       ₹14L - ₹6L = ₹8 Lakhs

Payback Period:           ₹14L / (₹8L/12) = ~14 months
```

### 3-Year Projection

| Year | Cost | Savings | Net Benefit | Cumulative |
|------|------|---------|-------------|------------|
| 1 | ₹14L (impl) + ₹6L (ops) | ₹14L | -₹6L | -₹6L |
| 2 | ₹6.5L | ₹15L | +₹8.5L | +₹2.5L |
| 3 | ₹7L | ₹16L | +₹9L | +₹11.5L |

**3-Year ROI: ~80%**

---

## 6. Risk Assessment

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OCR accuracy below target for handwritten | Medium | High | GPT-4o Vision achieves 85%+ accuracy; human-in-the-loop for low confidence |
| API costs exceed budget | Medium | Medium | Hybrid approach (70% local); implement caching; set usage alerts |
| Client format changes | High | Low | Flexible parsing rules; version detection |
| Staff resistance to automation | Medium | Medium | Gradual rollout; retrain staff for review roles |
| System downtime | Low | High | Cloud redundancy; fallback to manual process |

### Risk-Adjusted Scenarios

| Scenario | Probability | Annual Savings |
|----------|-------------|----------------|
| Pessimistic (70% target) | 20% | ₹9-10 Lakhs |
| Base case (100% target) | 60% | ₹12-16 Lakhs |
| Optimistic (120% target) | 20% | ₹15-19 Lakhs |

---

## 7. Alternative Comparison

| Option | 3-Year TCO | Pros | Cons |
|--------|------------|------|------|
| Do nothing | ₹50L+ (growing) | No investment | Scaling issues, errors continue |
| Hire more staff | ₹60L+ | Quick to implement | Linear scaling, same error rate |
| Commercial IDP (ABBYY/Kofax) | ₹60-80L | Proven solution | High license fees, vendor lock-in |
| **Custom build (this)** | **₹30-40L** | Full control, lowest TCO | Development time |

**Commercial Pricing References:**
- ABBYY FlexiCapture: ₹15-25L/year license — ABBYY partner quotes
- Kofax: ₹20-35L/year — Enterprise pricing
- Nanonets: ₹3-8L/year — nanonets.com pricing

---

## 8. Recommendation

**Proceed with custom development.**

Rationale:
1. **Lowest TCO** — ₹30-40L over 3 years vs ₹60L+ for alternatives
2. **Hybrid approach** — Optimizes costs by using AI only when needed
3. **Full ownership** — No vendor lock-in, can modify for specific needs
4. **Proven technology** — pdfplumber, GPT-4o, Pydantic are production-ready

### Success Criteria

| Metric | Target | Timeline |
|--------|--------|----------|
| Processing accuracy (structured) | >95% | Month 1 |
| Processing accuracy (OCR) | >85% | Month 2 |
| Auto-approval rate | >70% | Month 3 |
| Cost reduction | >50% | Month 6 |

---

## 9. Team Requirements

### Implementation Team (3-4 months)

| Role | Count | Responsibilities |
|------|-------|------------------|
| AI/ML Engineer | 1 | Pipeline development, model integration, GPT-4o implementation |
| Backend Developer | 1 | API development, parsers, integration with existing systems |
| DevOps (part-time) | 0.5 | Cloud setup, CI/CD, monitoring |
| QA Engineer (part-time) | 0.5 | Testing, validation, edge case handling |

### Required Skills

**AI/ML Engineer:**
- Python (pandas, pydantic)
- OpenAI API / GPT-4o Vision
- Document parsing (pdfplumber, python-docx)
- NLP basics, prompt engineering

**Backend Developer:**
- Python, REST APIs
- Integration experience (ERP/Tally connectors)
- Database design

### Post-Launch Support (Ongoing)

| Role | Allocation | Focus |
|------|------------|-------|
| AI Engineer | 0.25 FTE | Model monitoring, accuracy improvements |
| Support | 0.25 FTE | Issue resolution, client format changes |

---

## References

**Salary Data:**
- Glassdoor India: glassdoor.co.in/Salaries
- Naukri.com: naukri.com/salary
- Indeed India: indeed.co.in/salary
- LinkedIn Salary Insights India

**Technology Pricing:**
- OpenAI API: platform.openai.com/pricing (Dec 2025)
- AWS Calculator: calculator.aws

**Industry Research:**
- Gartner IDP Market Guide 2025
- McKinsey Operations Studies (manual data entry error rates: 5-10%)
- NASSCOM AI Adoption Reports

---

*Document Automation & Data Harmonization System — Business Case*
