// src/tagger.js
// Semantic intent tagger — understands casual, natural-language phrasing beyond strict key lookup.
// Fully local/offline. No external APIs.

import aliasMap from "./aliasMap.json";

// ─────────────────────────────────────────────────────────────────────────────
// 1. PATTERN-BASED SEMANTIC RULES
// Each rule has a regex and the tag(s) to assign when it matches.
// Patterns are tested against the full lowercased input.
// ─────────────────────────────────────────────────────────────────────────────
const semanticPatterns = [
  // ── Vehicles & Transport ─────────────────────────────────────────────────
  { pattern: /\b(my\s+)?(bike|scooty|car|vehicle|scooter|cycle|ev|moped|auto|cab|truck|van)\b/, tags: ["#vehicle"] },
  { pattern: /\b(parked|parking|park)\s+(my|the)?\s*\w+/, tags: ["#parking"] },
  { pattern: /\b(fuel|petrol|diesel|filled|pump)\b/, tags: ["#fuel"] },
  { pattern: /\b(number\s*plate|license\s*plate|registration|rc\s*book)\b/, tags: ["#vehicle", "#document"] },
  { pattern: /\b(drove|riding|drive|ride)\b/, tags: ["#vehicle"] },

  // ── Security / Credentials ────────────────────────────────────────────────
  { pattern: /\b(password|passwd|pwd|pass|passcode)\b/, tags: ["#security"] },
  { pattern: /\b(pin|mpin|atm\s*pin|otp)\b/, tags: ["#security"] },
  { pattern: /\b(wifi|wi-fi|network|hotspot)\s*(password|key|code|name)?\b/, tags: ["#security"] },
  { pattern: /\b(locker|lock\s*code|safe\s*code|combination)\b/, tags: ["#security"] },
  { pattern: /\b(secret|credentials|login|username)\b/, tags: ["#security"] },
  // "My door code is..." / "locker number is..."
  { pattern: /\b(door|room|gate|cabinet|cupboard)\s*(code|key|number|pin|password)\b/, tags: ["#security"] },

  // ── Finance ───────────────────────────────────────────────────────────────
  { pattern: /\b(account\s*number|acc\s*no|bank\s*acc)\b/, tags: ["#finance"] },
  { pattern: /\b(ifsc|swift|iban|micr)\b/, tags: ["#finance"] },
  { pattern: /\b(gpay|google\s*pay|phonepe|paytm|upi\s*id|upi)\b/, tags: ["#finance"] },
  { pattern: /\b(emi|instalment|loan|borrowed|lent|lend|borrow)\b/, tags: ["#finance"] },
  { pattern: /\b(salary|income|earned|transfer|credited|debited)\b/, tags: ["#finance"] },
  { pattern: /\b(invest|stock|share|mutual\s*fund|fd|fixed\s*deposit|ppf|sip)\b/, tags: ["#finance"] },
  { pattern: /\b(tax|gst|itr|return\s*file)\b/, tags: ["#finance"] },
  // Casual: "paid X rupees", "owe X", "got paid"
  { pattern: /\b(paid|owe|owes|owed|got\s*paid|send\s*money|receive\s*money)\b/, tags: ["#finance"] },
  // Large number + currency context
  { pattern: /₹[\d,]+|\brs\.?\s*\d+|\brupees?\b/, tags: ["#finance"] },

  // ── Bills / Subscriptions ─────────────────────────────────────────────────
  { pattern: /\b(electricity|light\s*bill|power\s*bill|eb\s*bill)\b/, tags: ["#bill"] },
  { pattern: /\b(water\s*bill|sewage|maintenance\s*fee|society\s*fee)\b/, tags: ["#bill"] },
  { pattern: /\b(internet\s*bill|broadband|jio|airtel|bsnl|vi|vodafone)\b/, tags: ["#bill"] },
  { pattern: /\b(recharge|mobile\s*recharge|prepaid|postpaid|plan)\b/, tags: ["#bill"] },
  { pattern: /\b(netflix|hotstar|prime|spotify|youtube\s*premium|subscription)\b/, tags: ["#bill"] },
  { pattern: /\b(due\s*date|last\s*date|pay\s*before|deadline)\b/, tags: ["#bill", "#reminder"] },
  // Casual: "have to pay..." / "pending bill"
  { pattern: /\b(pending\s*bill|unpaid|due)\b/, tags: ["#bill"] },

  // ── Documents & IDs ───────────────────────────────────────────────────────
  { pattern: /\b(aadhaar|aadhar)\b/, tags: ["#document"] },
  { pattern: /\b(pan\s*card|pan\s*number)\b/, tags: ["#document"] },
  { pattern: /\b(passport|visa|stamp)\b/, tags: ["#document"] },
  { pattern: /\b(driving\s*licen[sc]e|dl\s*number)\b/, tags: ["#document", "#vehicle"] },
  { pattern: /\b(voter\s*id|election\s*card)\b/, tags: ["#document"] },
  { pattern: /\b(birth\s*certificate|degree|marksheet|result|transcript)\b/, tags: ["#document"] },
  { pattern: /\b(kyc|verification|proof\s*of)\b/, tags: ["#document"] },

  // ── Insurance / Policy ────────────────────────────────────────────────────
  { pattern: /\b(insurance|policy\s*number|policy\s*no|insured)\b/, tags: ["#policy"] },
  { pattern: /\b(premium|claim|nominee|coverage|maturity)\b/, tags: ["#policy"] },
  { pattern: /\b(term\s*plan|health\s*plan|mediclaim|life\s*insurance)\b/, tags: ["#policy"] },

  // ── Health & Medicine ─────────────────────────────────────────────────────
  { pattern: /\b(medicine|tablet|capsule|syrup|injection|dose|dosage)\b/, tags: ["#health"] },
  { pattern: /\b(doctor|physician|specialist|hospital|clinic|pharmacy)\b/, tags: ["#health"] },
  { pattern: /\b(prescription|rx|prescription)\b/, tags: ["#health"] },
  { pattern: /\b(blood\s*test|urine\s*test|xray|x-ray|mri|scan|report)\b/, tags: ["#health"] },
  { pattern: /\b(blood\s*pressure|bp|sugar|diabetes|cholesterol|thyroid)\b/, tags: ["#health"] },
  { pattern: /\b(appointment|checkup|check-up|consultation)\b/, tags: ["#health", "#reminder"] },
  { pattern: /\b(allergy|allergic|side\s*effect)\b/, tags: ["#health"] },
  // Casual: "feeling sick", "not well", "gym", "workout"
  { pattern: /\b(feeling\s*sick|not\s*well|unwell|ill|headache|fever|cold)\b/, tags: ["#health"] },
  { pattern: /\b(gym|workout|exercise|run|jog|yoga|calories|weight\s*loss)\b/, tags: ["#health"] },

  // ── Travel ────────────────────────────────────────────────────────────────
  { pattern: /\b(flight|airline|airport|terminal|gate|boarding)\b/, tags: ["#travel"] },
  { pattern: /\b(train|irctc|pnr|railway|platform|station)\b/, tags: ["#travel"] },
  { pattern: /\b(bus|ksrtc|gsrtc|msrtc|redbus|seat\s*number)\b/, tags: ["#travel"] },
  { pattern: /\b(hotel|hostel|resort|check-?in|check-?out|booking\s*id)\b/, tags: ["#travel"] },
  { pattern: /\b(trip|vacation|holiday|tour|itinerary)\b/, tags: ["#travel"] },
  // Casual: "going to X", "traveling to X", "leaving for X"
  { pattern: /\b(going\s*to|travelling\s*to|traveling\s*to|leaving\s*for|reaching)\b/, tags: ["#travel"] },
  { pattern: /\b(luggage|baggage|passport|visa)\b/, tags: ["#travel"] },

  // ── Reminders & Tasks ────────────────────────────────────────────────────
  { pattern: /\b(remind|reminder|remember|don't\s*forget|don't forget|memo)\b/, tags: ["#reminder"] },
  { pattern: /\b(todo|to-do|to\s*do|task|errand|follow(\s*|-)?up)\b/, tags: ["#reminder"] },
  { pattern: /\b(meeting|call\s*at|appointment\s*at|event)\b/, tags: ["#reminder"] },
  { pattern: /\b(tomorrow|tonight|by\s*\d|at\s*\d+\s*(am|pm)|next\s*week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/, tags: ["#reminder"] },
  // Casual: "need to...", "have to...", "must..."
  { pattern: /\b(need\s*to|have\s*to|must|should|gotta|gonna)\b/, tags: ["#reminder"] },
  { pattern: /\b(deadline|due\s*on|submit|submit\s*by)\b/, tags: ["#reminder"] },

  // ── Shopping ─────────────────────────────────────────────────────────────
  { pattern: /\b(buy|purchase|order|get\s*me|pick\s*up|shop|shopping)\b/, tags: ["#shopping"] },
  { pattern: /\b(grocery|vegetables|fruits|milk|eggs|bread)\b/, tags: ["#shopping"] },
  { pattern: /\b(amazon|flipkart|myntra|meesho|swiggy\s*instamart|blinkit|zepto|bigbasket)\b/, tags: ["#shopping"] },
  { pattern: /\b(cart|wishlist|delivery|tracking|order\s*id|shipment)\b/, tags: ["#shopping"] },

  // ── Contacts & Addresses ─────────────────────────────────────────────────
  { pattern: /\b(address|flat\s*no|apartment|building|street|locality|area|colony|nagar|layout)\b/, tags: ["#contact"] },
  { pattern: /\b(phone\s*number|mobile\s*number|contact\s*number|call\s*\w+\s*at)\b/, tags: ["#contact"] },
  { pattern: /\b(pincode|zip\s*code|postal\s*code)\b/, tags: ["#contact"] },
  { pattern: /\b(email\s*id|gmail|yahoo|outlook|mail\s*id)\b/, tags: ["#contact"] },

  // ── Number pattern heuristics ─────────────────────────────────────────────
  // 10-digit number → likely phone number
  { pattern: /\b[6-9]\d{9}\b/, tags: ["#contact"] },
  // 16-digit number → card number
  { pattern: /\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b/, tags: ["#security"] },
  // 4-6 digit number near "pin/code/otp" (matched by pin/code patterns above)
  // IFSC: 4 letters + 7 alphanumeric
  { pattern: /\b[A-Z]{4}0[A-Z0-9]{6}\b/i, tags: ["#finance"] },
  // UPI ID format: something@bank
  { pattern: /\b\w+@(okaxis|okhdfcbank|okicici|oksbi|ybl|axl|upi|paytm|ibl)\b/i, tags: ["#finance"] },
  // PNR: 10-digit IRCTC
  { pattern: /\bpnr\s*:?\s*\d{10}\b/i, tags: ["#travel"] },

  // ── Casual / Intent phrasing ───────────────────────────────────────────────
  // "where did I put..." / "where is my..."
  { pattern: /\bwhere\s*(did\s*i\s*(put|keep|leave|park)|is\s*my|are\s*my)\b/, tags: ["#reminder"] },
  // "I lost my..." / "can't find my..."
  { pattern: /\b(lost|can'?t\s*find|misplaced|forgot\s*where)\b/, tags: ["#reminder"] },
  // "store this" / "save this" / "note this down"
  { pattern: /\b(store\s*this|save\s*this|note\s*this|keep\s*this|note\s*down|jot\s*this)\b/, tags: ["#reminder"] },
];

// ─────────────────────────────────────────────────────────────────────────────
// 2. STRICT ALIAS MAP LOOKUP (word-by-word, existing behaviour)
// ─────────────────────────────────────────────────────────────────────────────
function aliasLookup(text) {
  const words = text
    .toLowerCase()
    .replace(/[^a-zA-Z0-9#\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);

  const found = new Set();
  words.forEach((w) => {
    if (w.startsWith("#")) found.add(w);
    else if (aliasMap[w]) found.add(aliasMap[w]);
  });
  return found;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. CONTEXT CLUES — resolve ambiguous fragments using surrounding words
// ─────────────────────────────────────────────────────────────────────────────
function contextualTags(lowerText) {
  const found = new Set();

  // "my doctor said take X" → #health even if X isn't a medicine name
  if (/\bdoctor\s+said\b/.test(lowerText)) found.add("#health");

  // "forgot my X" — classify X contextually
  const forgotMatch = lowerText.match(/forgot\s+(?:my\s+)?(\w+)/);
  if (forgotMatch) {
    const noun = forgotMatch[1];
    if (/pass|pin|code|key|lock/.test(noun)) found.add("#security");
    else if (/card|wallet|cash|money/.test(noun)) found.add("#finance");
    else found.add("#reminder");
  }

  // "meeting with X at Y" / "call with X" → #reminder + contextual
  if (/\b(meeting|call)\s+with\b/.test(lowerText)) found.add("#reminder");

  // "owe X money" / "X owes me" → #finance
  if (/\bowe[sd]?\b/.test(lowerText)) found.add("#finance");

  // "visiting X hospital/clinic" → #health
  if (/\bvisiting\b.*\b(hospital|clinic|doctor)\b/.test(lowerText)) found.add("#health");

  // "parked at/near X" → #parking
  if (/\bparked\s+(at|near|in|outside|behind)\b/.test(lowerText)) found.add("#parking");

  // Numbers 4-6 digits near security context words
  if (/\b\d{4,6}\b/.test(lowerText) && /\b(pin|code|otp|password|locker)\b/.test(lowerText)) {
    found.add("#security");
  }

  return found;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. MAIN EXPORT: deriveTags — fully semantic tagger
// Combines alias lookup + pattern rules + contextual clues.
// Returns deduped sorted tag array.
// ─────────────────────────────────────────────────────────────────────────────
export function deriveTags(text) {
  if (!text || !text.trim()) return ["#note"];

  const lowerText = text.toLowerCase();
  const allTags = new Set();

  // A. Strict alias map lookup
  aliasLookup(text).forEach((t) => allTags.add(t));

  // B. Pattern-based semantic rules
  semanticPatterns.forEach(({ pattern, tags }) => {
    if (pattern.test(lowerText)) {
      tags.forEach((t) => allTags.add(t));
    }
  });

  // C. Context clue analysis
  contextualTags(lowerText).forEach((t) => allTags.add(t));

  // D. Fallback
  if (allTags.size === 0) allTags.add("#note");

  return Array.from(allTags).sort();
}
