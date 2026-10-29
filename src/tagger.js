// src/tagger.js
// Semantic intent tagger — understands natural-language phrasing beyond strict key lookup.
// Fully local/offline. No external APIs.

import aliasMap from "./aliasMap.json";

// ─────────────────────────────────────────────────────────────────────────────
// 1. PATTERN-BASED SEMANTIC RULES
// ─────────────────────────────────────────────────────────────────────────────
const semanticPatterns = [
  // ── Movies, Entertainment & Events ───────────────────────────────────────
  { pattern: /\b(bookmyshow|movie|cinema|multiplex|pvr|inox|cinepolis|theatre|theater|imax)\b/, tags: ["#entertainment", "#movie", "#ticket"] },
  { pattern: /\b(ticket|tickets|seat\s*[a-z]?\d+|audi\s*\d+|screen\s*\d+|show\s*time)\b/, tags: ["#ticket"] },
  { pattern: /\b(concert|live\s*show|standup|comedy|gig|festival)\b/, tags: ["#entertainment", "#event"] },

  // ── Vehicles & Transport (strict word boundary, ignore ticket seats/codes) ─
  { pattern: /\b(my\s+)?(bike|motorcycle|scooty|scooter|car|automobile|moped|auto\s+rickshaw|truck|bus)\b/, tags: ["#vehicle"] },
  { pattern: /\b(parked|parking)\s+(my|the)?\s*(car|bike|vehicle|scooter|spot|slot)?\b/, tags: ["#parking", "#vehicle"] },
  { pattern: /\b(fuel|petrol|diesel|cng)\s+(filled|pump|bill|receipt)?\b/, tags: ["#fuel", "#vehicle"] },
  { pattern: /\b(license\s*plate|registration\s*plate|rc\s*book|rc\s*card)\b/, tags: ["#vehicle", "#document"] },

  // ── Gadgets & Tech ───────────────────────────────────────────────────────
  { pattern: /\b(laptop|macbook|thinkpad|notebook|computer|pc|desktop)\b/, tags: ["#electronics", "#tech", "#laptop"] },
  { pattern: /\b(keyboard|mechanical\s*keyboard|keypad|typing)\b/, tags: ["#electronics", "#tech", "#keyboard"] },
  { pattern: /\b(mouse|trackpad|mousepad)\b/, tags: ["#electronics", "#tech"] },
  { pattern: /\b(headphone|headphones|earphones|airpods|earbuds)\b/, tags: ["#electronics", "#audio"] },
  { pattern: /\b(charger|cable|adapter|usb|hdmi|powerbank)\b/, tags: ["#electronics", "#tech"] },
  { pattern: /\b(backpack|bag|handbag|rucksack|schoolbag|officebag|luggage|suitcase)\b/, tags: ["#bag", "#accessory"] },

  // ── Festivals & Celebrations ─────────────────────────────────────────────
  { pattern: /\b(diwali|deepavali|festival|festive|crackers|diya|lights)\b/, tags: ["#festival", "#celebration"] },
  { pattern: /\b(birthday|bday|anniversary|party)\b/, tags: ["#celebration", "#reminder"] },
  { pattern: /\b(christmas|xmas|new\s*year|eid|holi)\b/, tags: ["#festival", "#celebration"] },

  // ── Photos & Selfies ──────────────────────────────────────────────────────
  { pattern: /\b(selfie|my\s+face|photo\s+of\s+me|my\s+selfie)\b/, tags: ["#photo", "#selfie"] },

  // ── Security / Credentials ────────────────────────────────────────────────
  { pattern: /\b(password|passwd|pwd|passcode)\b/, tags: ["#security"] },
  { pattern: /\b(pin|mpin|atm\s*pin|otp)\b/, tags: ["#security"] },
  { pattern: /\b(wifi|wi-fi|hotspot)\b[\s\S]{0,80}\b(password|key|code|name)\b|\bnetwork\s+(password|key|code)\b/, tags: ["#security"] },
  { pattern: /\b(locker|lock\s*code|safe\s*code|combination)\b/, tags: ["#security"] },
  { pattern: /\b(secret|credentials|login|username)\b/, tags: ["#security"] },

  // ── Finance ───────────────────────────────────────────────────────────────
  { pattern: /\b(account\s*number|acc\s*no|bank\s*acc|credit\s*card|debit\s*card|bank\s*card)\b/, tags: ["#finance"] },
  { pattern: /\b(ifsc|swift|iban|micr)\b/, tags: ["#finance"] },
  { pattern: /\b(gpay|google\s*pay|phonepe|paytm|upi\s*id|upi)\b/, tags: ["#finance"] },
  { pattern: /\b(emi|instalment|loan|borrowed|lent)\b/, tags: ["#finance"] },
  { pattern: /\b(salary|income|transfer|credited|debited)\b/, tags: ["#finance"] },
  { pattern: /\b(invest|mutual\s*fund|fixed\s*deposit|ppf|sip)\b/, tags: ["#finance"] },
  { pattern: /₹[\d,]+|\brs\.?\s*\d+|\brupees?\b/, tags: ["#finance"] },

  // ── Bills & Invoices ──────────────────────────────────────────────────────
  { pattern: /\b(electricity\s*bill|light\s*bill|power\s*bill|eb\s*bill|water\s*bill|gas\s*bill)\b/, tags: ["#bill"] },
  { pattern: /\b(invoice|tax\s*invoice|receipt|bill\s+amount|bill\s+due)\b/, tags: ["#bill"] },
  { pattern: /\b(recharge|mobile\s*recharge|prepaid|postpaid)\b/, tags: ["#bill"] },
  { pattern: /\b(netflix|hotstar|prime\s*video|spotify|subscription)\b/, tags: ["#bill", "#subscription"] },

  // ── Government Documents & Identity ──────────────────────────────────────
  { pattern: /\b(aadhaar|aadhar|uidai)\b/, tags: ["#document"] },
  { pattern: /\b(pan\s*card|permanent\s*account\s*number)\b/, tags: ["#document"] },
  { pattern: /\b(passport|visa\s*stamp)\b/, tags: ["#document"] },
  { pattern: /\b(driving\s*licen[sc]e|learner\s*licen[sc]e)\b/, tags: ["#document", "#vehicle"] },
  { pattern: /\b(voter\s*id|election\s*commission)\b/, tags: ["#document"] },
  { pattern: /\b(marksheet|degree\s*certificate|diploma|birth\s*certificate)\b/, tags: ["#document"] },

  // ── Health & Medicine ────────────────────────────────────────────────────
  { pattern: /\b(medicine|tablet|capsule|syrup|dosage|dose)\b/, tags: ["#health"] },
  { pattern: /\b(doctor|physician|hospital|clinic|pharmacy)\b/, tags: ["#health"] },
  { pattern: /\b(prescription|rx|lab\s*report|blood\s*test|x-ray|mri)\b/, tags: ["#health"] },

  // ── Travel ────────────────────────────────────────────────────────────────
  { pattern: /\b(flight|airline|airport|boarding\s*pass)\b/, tags: ["#travel"] },
  { pattern: /\b(train|irctc|pnr|railway|railway\s*station)\b/, tags: ["#travel"] },
  { pattern: /\b(hotel|hostel|resort|check-?in|booking\s*id)\b/, tags: ["#travel"] },

  // ── Reminders & Tasks ────────────────────────────────────────────────────
  { pattern: /\b(remind|reminder|remember|don't\s*forget|memo)\b/, tags: ["#reminder"] },
  { pattern: /\b(todo|to-do|to\s*do|follow(\s*|-)?up)\b/, tags: ["#reminder"] },
  { pattern: /\b(meeting|appointment\s*at)\b/, tags: ["#reminder"] },

  // ── Shopping & E-Commerce ─────────────────────────────────────────────────
  { pattern: /\b(order\s*placed|order\s*confirmed|tracking\s*id|delivery\s*partner)\b/, tags: ["#shopping"] },
  { pattern: /\b(amazon|flipkart|myntra|swiggy|zomato|blinkit|zepto|instamart)\b/, tags: ["#shopping"] },
  { pattern: /\b(grocery|vegetables|fruits|supermarket)\b/, tags: ["#shopping", "#grocery"] },
];

// Words that should NOT trigger random generic tags when occurring inside OCR text
const STOPWORDS = new Set([
  "the", "and", "for", "with", "from", "that", "this", "have", "will", "your",
  "about", "show", "book", "order", "view", "edit", "call", "send", "date", "time",
  "card", "mode", "text", "pure", "image", "photo", "screen", "live", "near", "open"
]);

function aliasLookup(text) {
  const words = text
    .toLowerCase()
    .replace(/[^a-zA-Z0-9#\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w));

  const found = new Set();
  words.forEach((w) => {
    if (w.startsWith("#")) found.add(w);
    else if (aliasMap[w]) {
      // Avoid false vehicle tag from single isolated generic words
      if (aliasMap[w] === "#vehicle" && ["auto", "driving", "registration", "rc"].includes(w)) {
        return;
      }
      found.add(aliasMap[w]);
    }
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

  BLACKLIST.forEach((b) => allTags.delete(b));

  return Array.from(allTags).sort();
}
