// src/tagger.js
// Semantic intent tagger — understands casual, natural-language phrasing beyond strict key lookup.
// Fully local/offline. No external APIs.

import aliasMap from "./aliasMap.json";

// ─────────────────────────────────────────────────────────────────────────────
// 1. PATTERN-BASED SEMANTIC RULES
// ─────────────────────────────────────────────────────────────────────────────
const semanticPatterns = [
  // ── Vehicles & Transport ─────────────────────────────────────────────────
  { pattern: /\b(my\s+)?(bike|scooty|car|vehicle|scooter|cycle|ev|moped|auto|cab|truck|van|twowheeler|fourwheeler)\b/, tags: ["#vehicle"] },
  { pattern: /\b(parked|parking|park)\s+(my|the)?\s*\w+/, tags: ["#parking"] },
  { pattern: /\b(fuel|petrol|diesel|filled|pump)\b/, tags: ["#fuel"] },
  { pattern: /\b(number\s*plate|license\s*plate|registration|rc\s*book|plate)\b/, tags: ["#vehicle", "#document"] },
  { pattern: /\b(drove|riding|drive|ride)\b/, tags: ["#vehicle"] },
  // Match standard vehicle registration formats (e.g. BR01, DL04, MH12, etc.)
  { pattern: /\b[A-Z]{2}\s*\d{1,2}\s*[A-Z]{0,3}\s*\d{3,4}\b/i, tags: ["#vehicle"] },
  // Match Bharat (BH) series format: e.g. 22 BH 6517 A
  { pattern: /\b\d{2}\s*BH\s*\d{2,4}\s*[A-Z]{1,2}\b/i, tags: ["#vehicle"] },

  // ── Festivals & Events ───────────────────────────────────────────────────
  { pattern: /\b(diwali|deepavali|festival|festive|crackers|diya|lights)\b/, tags: ["#festival", "#celebration"] },
  { pattern: /\b(birthday|bday|anniversary|party)\b/, tags: ["#celebration", "#reminder"] },
  { pattern: /\b(christmas|xmas|new\s*year|eid|holi)\b/, tags: ["#festival", "#celebration"] },

  // ── Photos & Selfies ──────────────────────────────────────────────────────
  { pattern: /\b(selfie|photo|picture|pic|pics|portrait|snap|my\s+face|photo\s+of\s+me|my\s+selfie|my\s+photo|camera\s+roll)\b/, tags: ["#photo", "#selfie"] },

  // ── Security / Credentials ────────────────────────────────────────────────
  { pattern: /\b(password|passwd|pwd|pass|passcode)\b/, tags: ["#security"] },
  { pattern: /\b(pin|mpin|atm\s*pin|otp)\b/, tags: ["#security"] },
  { pattern: /\b(wifi|wi-fi|network|hotspot)\s*(password|key|code|name)?\b/, tags: ["#security"] },
  { pattern: /\b(locker|lock\s*code|safe\s*code|combination)\b/, tags: ["#security"] },
  { pattern: /\b(secret|credentials|login|username)\b/, tags: ["#security"] },
  { pattern: /\b(door|room|gate|cabinet|cupboard)\s*(code|key|number|pin|password)\b/, tags: ["#security"] },

  // ── Finance ───────────────────────────────────────────────────────────────
  { pattern: /\b(account\s*number|acc\s*no|bank\s*acc)\b/, tags: ["#finance"] },
  { pattern: /\b(ifsc|swift|iban|micr)\b/, tags: ["#finance"] },
  { pattern: /\b(gpay|google\s*pay|phonepe|paytm|upi\s*id|upi)\b/, tags: ["#finance"] },
  { pattern: /\b(emi|instalment|loan|borrowed|lent|lend|borrow)\b/, tags: ["#finance"] },
  { pattern: /\b(salary|income|earned|transfer|credited|debited)\b/, tags: ["#finance"] },
  { pattern: /\b(invest|stock|share|mutual\s*fund|fd|fixed\s*deposit|ppf|sip)\b/, tags: ["#finance"] },
  { pattern: /\b(tax|gst|itr|return\s*file)\b/, tags: ["#finance"] },
  { pattern: /\b(paid|owe|owes|owed|got\s*paid|send\s*money|receive\s*money)\b/, tags: ["#finance"] },
  { pattern: /₹[\d,]+|\brs\.?\s*\d+|\brupees?\b/, tags: ["#finance"] },

  // ── Bills / Subscriptions ─────────────────────────────────────────────────
  { pattern: /\b(electricity|light\s*bill|power\s*bill|eb\s*bill)\b/, tags: ["#bill"] },
  { pattern: /\b(water\s*bill|sewage|maintenance\s*fee|society\s*fee)\b/, tags: ["#bill"] },
  { pattern: /\b(internet\s*bill|broadband|jio|airtel|bsnl|vi|vodafone)\b/, tags: ["#bill"] },
  { pattern: /\b(recharge|mobile\s*recharge|prepaid|postpaid|plan)\b/, tags: ["#bill"] },
  { pattern: /\b(netflix|hotstar|prime|spotify|youtube\s*premium|subscription)\b/, tags: ["#bill"] },
  { pattern: /\b(due\s*date|last\s*date|pay\s*before|deadline)\b/, tags: ["#bill", "#reminder"] },
  { pattern: /\b(pending\s*bill|unpaid|due)\b/, tags: ["#bill"] },

  // ── Documents & IDs ───────────────────────────────────────────────────────
  { pattern: /\b(aadhaar|aadhar)\b/, tags: ["#document"] },
  { pattern: /\b(pan\s*card|pan\s*number)\b/, tags: ["#document"] },
  { pattern: /\b(passport|visa|stamp)\b/, tags: ["#document"] },
  { pattern: /\b(driving\s*licen[sc]e|dl\s*number)\b/, tags: ["#document", "#vehicle"] },
  { pattern: /\b(voter\s*id|election\s*card)\b/, tags: ["#document"] },
  { pattern: /\b(birth\s*certificate|degree|marksheet|result|transcript)\b/, tags: ["#document"] },
  { pattern: /\b(kyc|verification|proof\s*of)\b/, tags: ["#document"] },

  // ── Health & Medicine ─────────────────────────────────────────────────────
  { pattern: /\b(medicine|tablet|capsule|syrup|injection|dose|dosage)\b/, tags: ["#health"] },
  { pattern: /\b(doctor|physician|specialist|hospital|clinic|pharmacy)\b/, tags: ["#health"] },
  { pattern: /\b(prescription|rx)\b/, tags: ["#health"] },
  { pattern: /\b(blood\s*test|urine\s*test|xray|x-ray|mri|scan|report)\b/, tags: ["#health"] },
  { pattern: /\b(blood\s*pressure|bp|sugar|diabetes|cholesterol|thyroid)\b/, tags: ["#health"] },
  { pattern: /\b(appointment|checkup|check-up|consultation)\b/, tags: ["#health", "#reminder"] },
  { pattern: /\b(allergy|allergic|side\s*effect)\b/, tags: ["#health"] },

  // ── Travel ────────────────────────────────────────────────────────────────
  { pattern: /\b(flight|airline|airport|terminal|gate|boarding)\b/, tags: ["#travel"] },
  { pattern: /\b(train|irctc|pnr|railway|platform|station)\b/, tags: ["#travel"] },
  { pattern: /\b(hotel|hostel|resort|check-?in|check-?out|booking\s*id)\b/, tags: ["#travel"] },
  { pattern: /\b(trip|vacation|holiday|tour|itinerary)\b/, tags: ["#travel"] },

  // ── Reminders & Tasks ────────────────────────────────────────────────────
  { pattern: /\b(remind|reminder|remember|don't\s*forget|memo)\b/, tags: ["#reminder"] },
  { pattern: /\b(todo|to-do|to\s*do|task|errand|follow(\s*|-)?up)\b/, tags: ["#reminder"] },
  { pattern: /\b(meeting|call\s*at|appointment\s*at|event)\b/, tags: ["#reminder"] },
  { pattern: /\b(tomorrow|tonight|next\s*week)\b/, tags: ["#reminder"] },

  // ── Shopping ─────────────────────────────────────────────────────────────
  { pattern: /\b(buy|purchase|order|pick\s*up|shop|shopping)\b/, tags: ["#shopping"] },
  { pattern: /\b(grocery|vegetables|fruits|milk|eggs|bread)\b/, tags: ["#shopping"] },
  { pattern: /\b(amazon|flipkart|swiggy|blinkit|zepto)\b/, tags: ["#shopping"] },

  // ── Contacts & Addresses ─────────────────────────────────────────────────
  { pattern: /\b(address|flat\s*no|apartment|building|street|area|colony)\b/, tags: ["#contact"] },
  { pattern: /\b(phone\s*number|mobile\s*number|contact\s*number)\b/, tags: ["#contact"] },
  { pattern: /\b(pincode|zip\s*code|postal\s*code)\b/, tags: ["#contact"] },
];

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

export function deriveTags(text) {
  const BLACKLIST = new Set(["#washroom", "#bathroom", "#toilet"]);
  if (!text || !text.trim()) return [];

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

  // Remove any blacklisted tags
  BLACKLIST.forEach((b) => allTags.delete(b));

  return Array.from(allTags).sort();
}
  if (!text || !text.trim()) return [];

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

  return Array.from(allTags).sort();
}
