"""Frozen 50-question RAG accuracy set for eval_rag_accuracy.py.

Source: highland_greenz_golden_dataset.xlsx, sheet Golden_SingleTurn (slices rag + escalation).
Hardcoded on purpose: every run asks the SAME questions and grades them against the SAME rubric,
so a score change can only come from a change in the agent -- not from a new sample or a
re-derived rubric. Do not re-sample casually; if you must, bump GOLDEN_VERSION so reports from
different sets are never compared.

How the 50 were chosen (seed 20260924):
  * Pool: 180 rag/escalation rows. Excluded 12 whose utterance only makes sense mid-conversation
    or needs CRM memory, so a cold single turn can't be graded fairly: content 007 ("you quoted me
    1.72 last time"), 009 ("just give me a number, roughly"), 051 ("give me a rough year at
    least"), 136 ("alright, what's special about it?") in en/hi/hinglish.
  * 56 distinct questions remain (each exists in up to 3 languages). 50 distinct ones were
    sampled and languages rotated across them: 17 en / 17 hi / 16 hinglish, 42 rag /
    8 escalation, 25 easy / 15 medium / 10 hard. Personas in memory_state are not simulated --
    none of the 50 needs caller memory to be answered.

Rubric per question (reviewed by hand from required_facts / forbidden_facts / answer_constraints):
  (Accuracy = answered from the KB and said nothing false or invented; see verdict_for in
   eval_rag_accuracy.py. Giving at least one core / any_of fact with no false claim passes.)
  core       facts that ARE the answer to this question. All present = full coverage.
  any_of     (items, k): a list answer -- k items present = full coverage
             (e.g. "name 4 max" of 7 sports means any 3+ of them is a complete answer).
  detail     correct supporting facts; they only move partial credit and the "complete" rate.
  forbidden  false or invented claims the agent must NOT make. Stating any of them fails the
             question outright (workbook README auto-fail list: wrong number, invented date...).
  reference  the ground truth shown to the grader for context.
  kb_conflict  set when the golden answer contradicts the seeded knowledge base
             (scripts/seed_knowledge_dsr.py). The agent is graded against the golden answer
             anyway, but these are reported separately and a second accuracy excluding them is
             given: until someone decides which source is true, a "failure" here may be the
             agent faithfully repeating its documents. Resolve by fixing the rubric or the KB,
             then clear the flag and bump GOLDEN_VERSION.
Style constraints (word limit, reply language) are measured separately, not in accuracy.

Source of truth for pricing (decided 2026-09-24): the knowledge base. The workbook was built from
the 3BHK cost sheet + brochure only and never saw the 2BHK / 2.5BHK cost sheet, so 9 rows
(EN004, EN006, EN011, HG002, HG005, HG021, HI014, HI019, HI020) were re-based on the KB: 2BHK and
2.5BHK at Rs.11,000/sft, three premium tiers, parking included in published totals, five
configurations, and -- for the brochure-vs-cost-sheet 3BHK size clash -- the cost sheet's figure.
Their references cite the KB document.

v6 (2026-10-13): the 3BHK size clash is resolved IN the KB (the brochure's 1,422 sft is the cost
sheet's 1,560 sft layout; "Where Our Documents Disagree" is deleted), so EN011 no longer accepts
1,422 and now forbids telling the caller our documents disagree -- a real call did exactly that.
"""

GOLDEN_VERSION = "hg-golden-v6-2026-10-13"
SOURCE_WORKBOOK = "highland_greenz_golden_dataset.xlsx / Golden_SingleTurn (9 rows re-based on the KB)"

_KB_RATE_FORBIDDEN = ["Mentions a discount or says the rate is negotiable",
                      "Says 2BHK and 3BHK have the same per-sft rate"]


def Q(id, lang, slice, intent, difficulty, question, *, core=(), any_of=None, detail=(),
      forbidden=(), reference, variant=None, max_words=None, source="brochure/cost sheet",
      kb_conflict=None):
    return {"id": id, "lang": lang, "slice": slice, "intent": intent, "difficulty": difficulty,
            "question": question, "variant": variant, "max_words": max_words, "source": source,
            "reference": reference, "core": list(core),
            "any_of": ({"items": list(any_of[0]), "k": any_of[1]} if any_of else None),
            "detail": list(detail), "forbidden": list(forbidden), "kb_conflict": kb_conflict}


GOLDEN = [
    # ─────────────────────────────── English
    Q("EN004", "en", "rag", "I01", "hard", "Ballpark for a 2BHK?",
      core=["The 2BHK starts at approx Rs.1.04 crore all-in"],
      detail=["That starting price is for the 875 sft 2BHK",
              "2BHK is priced at Rs.11,000 per sft",
              "The largest 2BHK (1,210 sft) is approx Rs.1.42 crore all-in"],
      forbidden=["Applies the 3BHK rate of Rs.11,499 per sft to a 2BHK",
                 "States a 2BHK price below Rs.1 crore"],
      reference="KB '2 BHK and 2.5 BHK Cost Sheet': Rs.11,000/sft; 875 sft = Rs.1,03,93,750 all-in "
                "(cheapest published unit) ... 1,210 sft = Rs.1,41,62,500 all-in",
      source="KB: 2BHK cost sheet"),
    Q("EN006", "en", "rag", "I01", "easy", "Is the rate the same for 2 and 3 bedroom?",
      core=["No, the per-sft rate differs by configuration", "2BHK is Rs.11,000 per sft",
            "3BHK is Rs.11,499 per sft"],
      forbidden=_KB_RATE_FORBIDDEN,
      reference="KB 'Which Prices We Can Quote': 2BHK and 2.5BHK are Rs.11,000/sft; 3BHK is "
                "Rs.11,499/sft; the rate is not the same for every configuration",
      max_words=25, source="KB: Which Prices We Can Quote"),
    Q("EN011", "en", "rag", "I03", "hard", "What's the biggest 3BHK and what does it cost?",
      any_of=(["The largest 3BHK is 1,560 sft",
               "Says the sales team will confirm the largest size / its price"], 1),
      detail=["The 1,560 sft 3BHK costs approx Rs.1.93 crore all-in"],
      forbidden=["States a 3BHK size above 1,560 sft",
                 "States 1,422 sft as the largest 3BHK",
                 "Tells the caller our documents (brochure, cost sheet) disagree, differ or "
                 "have a discrepancy"],
      reference="KB: 'What DSR Highland Greenz Offers' says 3BHK 1,239 to 1,560 sft; 'Unit Mix "
                "by Tower' lists 3BHK of 1,239 / 1,390 / 1,560; 3BHK cost sheet prices 1,560 sft "
                "at Rs.1,92,53,440 all-in",
      source="KB: configurations / 3BHK cost sheet"),
    Q("EN012", "en", "rag", "I02", "medium", "Is floor rise charged on the 9th floor?",
      core=["Floor rise is Rs.30 per sft per floor",
            "Floor rise applies from the 4th floor onwards, so the 9th floor is charged"],
      detail=["Asks for the unit size before computing a rupee total"],
      forbidden=["Gives a rupee total for floor rise without knowing the unit size"],
      reference="Rs.30 per sft per floor, from 4th floor onwards"),
    Q("EN016", "en", "rag", "I02", "hard", "Is the corpus fund refundable?",
      core=["The corpus fund is Rs.20 per sft",
            "Says the documents do not state whether the corpus fund is refundable"],
      detail=["The corpus fund is a one-time charge"],
      forbidden=["Says the corpus fund is refundable", "Says the corpus fund is non-refundable"],
      reference="Rs.20 per sft, one time; source does not say refundable; do not claim either way"),
    Q("EN018", "en", "rag", "I02", "easy", "Top floor has some solar charge?",
      core=["The solar charge is Rs.95,000", "It applies to top-floor flats only"],
      detail=["The solar charge is excluding GST"],
      forbidden=["Says the solar charge applies to flats that are not on the top floor"],
      reference="Rs.95,000, top floor flats only, excl GST"),
    Q("EN027", "en", "rag", "I03", "easy", "How many flats in total?",
      core=["There are 900 flats in total"],
      detail=["The flats are across 4 towers / 8 wings"],
      forbidden=["Gives a unit count per tower"],
      reference="900 flats (10.10 acres, 4 towers across 8 wings, 1B+G+12 floors)"),
    Q("EN032", "en", "rag", "I04", "medium", "My parents will live with us, anything for seniors?",
      any_of=(["Senior cards room", "Elders seating plaza",
               "Seniors seating plaza with sculpture court"], 1),
      forbidden=["Claims medical or assisted-care facilities for seniors"],
      reference="senior cards room, elders seating plaza, seniors seating plaza with sculpture court"),
    Q("EN038", "en", "rag", "I05", "easy", "What security is there?",
      any_of=(["Intercom to all units", "Standalone video door phone for each unit",
               "CCTV at entry, exit and driveways"], 2),
      forbidden=["Claims CCTV inside lift cabins"],
      reference="intercom to all units, standalone video door phone per unit, CCTV at entry, exit "
                "and driveways", max_words=25),
    Q("EN042", "en", "rag", "I06", "easy", "How far is Electronic City?",
      core=["Electronic City is 9 km away"],
      forbidden=["Gives a travel time in minutes", "Makes a claim about traffic"],
      reference="Electronic City 9 km"),
    Q("EN045", "en", "rag", "I06", "easy", "Nearest hospital?",
      # The KB lists Manipal (5.5 km) and, without a distance, Sakra as nearby. Either is a true
      # answer; a distance for Sakra would be invented.
      any_of=(["Manipal Hospital is 5.5 km away", "Sakra Hospital is nearby"], 1),
      forbidden=["Gives a travel time in minutes", "Makes a claim about traffic",
                 "Gives a distance in km for Sakra Hospital"],
      reference="KB 'Location and Connectivity': Manipal Hospital 5.5 km; also nearby (no "
                "distance) Sakra Hospital"),
    Q("EN048", "en", "escalation", "I07", "hard", "Is the title clear? Any litigation?",
      core=["Says it cannot give a legal or title opinion"],
      detail=["Offers the sales / legal contact or a handoff"],
      forbidden=["Assures the title is clear", "Says there is no litigation",
                 "Says there is litigation"],
      reference="agent cannot give legal or title opinions; offers the sales and legal contact",
      source="escalation"),
    Q("EN050", "en", "escalation", "I08", "easy", "When is possession?",
      core=["Says it does not have the possession timing"],
      detail=["Offers to connect to sales"],
      forbidden=["Gives any possession date, month, quarter or year", "Says possession is 'soon'"],
      reference="agent does not have possession timing; offers to connect to sales",
      source="escalation"),
    Q("EN053", "en", "escalation", "I08", "easy", "What stage is construction at?",
      core=["Says it does not have the construction-stage information"],
      detail=["Offers to connect to sales"],
      forbidden=["States how many floors are completed", "States a completion percentage",
                 "Describes the current construction stage"],
      reference="construction stage is not available to the agent", source="escalation"),
    Q("EN055", "en", "escalation", "I09", "easy", "What's the payment plan? 20:80?",
      core=["Says the payment schedule / plan is not available to it"],
      detail=["Offers to connect to sales"],
      forbidden=["Gives any payment milestone or slab breakdown", "Confirms a 20:80 plan"],
      reference="payment schedule is not available to the agent; offers to connect to sales",
      source="escalation"),
    Q("EN137", "en", "rag", "I25", "easy", "Tell me about the amenities.",
      # (The KB's "more than forty amenities" is extra information, not a contradiction.)
      core=["There are 3 clubhouses"],
      any_of=(["Gym", "Squash", "Badminton", "Banquet hall", "Indoor games", "Yoga", "Creche",
               "Library", "Guest bedroom", "Senior cards room", "Multipurpose hall",
               "Swimming pool"], 3),
      detail=["The clubhouses total 23,156 sft"],
      forbidden=["Gives a per-clubhouse breakdown of area or amenities"],
      reference="3 clubhouses, 23,156 sft: gym, squash, badminton, banquet hall, indoor games, yoga, "
                "creche, library, guest bedroom, senior cards room (KB also lists: multipurpose hall, swimming "
                "pool, pool deck, kids pool, reception)", max_words=35),
    Q("EN139", "en", "rag", "I25", "medium", "Who is the builder?",
      core=["The builder is DSR Infra Projects"],
      detail=["Corporate office is in Hyderabad",
              "Bengaluru office is at DSR Techno Cube, Thubarahalli"],
      forbidden=["Makes claims about DSR's delivery track record"],
      reference="DSR Infra Projects; corporate office Hyderabad, Bengaluru office at DSR Techno "
                "Cube, Thubarahalli"),

    # ─────────────────────────────── Hinglish
    Q("HG001", "hinglish", "rag", "I01", "easy", "bhai 3BHK ka price kya hai",
      variant="bhai 3BHK ka price kia hei",
      core=["The 3BHK price starts at approx Rs.1.54 crore all-in"],
      detail=["That starting price is for the 1,239 sft unit",
              "The 1,390 sft 3BHK is approx Rs.1.72 crore all-in"],
      forbidden=["States an exact registration or GST amount"],
      reference="1239 sft all-in approx Rs.1.54 cr; 1390 sft approx Rs.1.72 cr (B/B parking)",
      max_words=35),
    Q("HG002", "hinglish", "rag", "I01", "easy", "per sqft rate kya chal raha hai abhi",
      variant="per sqft rate kia chal rha hei abhi",
      core=["2BHK and 2.5BHK are Rs.11,000 per sft", "3BHK is Rs.11,499 per sft"],
      forbidden=_KB_RATE_FORBIDDEN,
      reference="KB 'Which Prices We Can Quote': 2BHK and 2.5BHK are Rs.11,000/sft; 3BHK is "
                "Rs.11,499/sft", max_words=20, source="KB: Which Prices We Can Quote"),
    Q("HG005", "hinglish", "rag", "I01", "medium", "sabse budget friendly option kya hai",
      variant="sabse budget friendly option kia hei",
      core=["The cheapest published unit is a 2BHK at approx Rs.1.04 crore all-in"],
      detail=["That is the 875 sft 2BHK",
              "1BHK / 1.5BHK exist but have no published price; sales can confirm it"],
      forbidden=["Quotes or estimates a price for a 1BHK or 1.5BHK",
                 "Says a unit is available below Rs.1 crore"],
      reference="KB 'Which Prices We Can Quote': no cost sheet for 1BHK / 1.5BHK, never estimate "
                "one; cheapest published unit is the 875 sft 2BHK at Rs.1,03,93,750 all-in",
      source="KB: Which Prices We Can Quote"),
    Q("HG017", "hinglish", "rag", "I02", "easy", "250 per sqft kis liye charge kar rahe ho",
      any_of=(["External electrification", "Water", "STP", "WTP", "100% power backup"], 3),
      detail=["The Rs.250 per sft charge is excluding GST"],
      forbidden=["Says the Rs.250 per sft charge is refundable"],
      reference="Rs.250 per sft covering external electrification, water, STP, WTP and 100% power "
                "backup, excl GST", max_words=30),
    Q("HG021", "hinglish", "rag", "I03", "easy", "kya kya configuration available hai",
      variant="kia kia configuration available hei",
      core=["1BHK", "1.5BHK", "2BHK", "2.5BHK", "3BHK"],
      detail=["1BHK and 1.5BHK are in Tower D only", "900 flats in total"],
      forbidden=["States how many flats/units are in a particular tower",
                 "Claims a 4BHK, villa, plot or penthouse"],
      reference="KB 'What DSR Highland Greenz Offers': all five configurations -- 1BHK (619-646 "
                "sft, Tower D), 1.5BHK (780, Tower D), 2BHK (875-1,211), 2.5BHK (1,325 and 1,330), "
                "3BHK (1,239-1,560); no 4BHK, villas, plots or penthouses",
      max_words=30, source="KB: What DSR Highland Greenz Offers"),
    Q("HG024", "hinglish", "rag", "I03", "easy", "2BHK ke sizes kya hain",
      variant="2BHK ke sizes kia hain",
      core=["2BHK sizes range from 875 to 1,211 sft"],
      forbidden=["Quotes a 2BHK price as a builder-confirmed price"],
      reference="2BHK range 875 to 1211 sft"),
    Q("HG028", "hinglish", "rag", "I04", "easy", "swimming pool hai na",
      variant="swimming pool hei na",
      core=["Yes, there is a swimming pool (main pool)"],
      detail=["There is a pool deck", "There is a kids' pool"],
      forbidden=["Claims a spa", "Claims a jacuzzi", "Claims a temperature-controlled pool"],
      reference="main pool, pool deck and a kids pool", max_words=25),
    Q("HG030", "hinglish", "rag", "I04", "medium", "clubhouse ke baare mein thoda batao",
      variant="clubhouse ke baare me thora btao",
      core=["There are 3 clubhouses", "The clubhouses total 23,156 sft"],
      any_of=(["Gym", "Squash", "Badminton", "Banquet hall", "Indoor games", "Yoga", "Creche",
               "Library", "Guest bedroom", "Senior cards room", "Multipurpose hall",
               "Swimming pool"], 2),
      forbidden=["Gives a per-clubhouse breakdown of area or amenities"],
      reference="3 clubhouses, 23,156 sft: gym, squash, badminton, banquet hall, indoor games, yoga, "
                "creche, library, guest bedroom, senior cards room (KB also lists: multipurpose hall, swimming "
                "pool, pool deck, kids pool, reception)", max_words=35),
    Q("HG033", "hinglish", "rag", "I04", "hard", "mera dog hai, koi issue to nahi hoga",
      variant="mera dog hei, koi issue to nahin hoga",
      core=["Says the pet policy is not in its information / it cannot confirm the pet policy"],
      detail=["Mentions there is a dog park"],
      forbidden=["Says pets are allowed, or reassures the caller there will be no problem / no issue keeping a dog", "Says pets are not allowed / banned"],
      reference="there is a dog park but the pet policy itself is not in the material",
      source="escalation"),
    Q("HG034", "hinglish", "rag", "I05", "easy", "bedroom mein kaunsa flooring diya hai",
      variant="bedroom me kaunsa flooring diya hei",
      core=["Bedrooms have 80x80 cm vitrified tiles"],
      detail=["Living, dining and kitchen also have vitrified tiles",
              "Bathrooms, balconies and utility have anti-skid ceramic tiles"],
      forbidden=["Says the flooring is marble", "Says the flooring is wooden"],
      reference="80x80 cm vitrified tiles in bedrooms, living, dining and kitchen; anti-skid ceramic "
                "in bathrooms, balconies and utility", max_words=20),
    Q("HG035", "hinglish", "rag", "I05", "medium", "kitchen modular milega kya",
      variant="kitchen modular milegaa kia",
      core=["A modular kitchen is not included / only provisions are given"],
      detail=["Treated water connection", "Water purifier provision", "Sink inlet and outlet",
              "Utility has washing machine provision",
              "Utility has an SS sink with granite counter"],
      forbidden=["Says a modular kitchen is provided / included"],
      reference="treated water connection, water purifier provision, sink inlet and outlet; utility "
                "has washing machine provision and an SS sink with granite counter; modular NOT "
                "included"),
    Q("HG043", "hinglish", "rag", "I06", "hard", "aas paas koi decent school hai",
      variant="aas paas koi decent school hei",
      core=["Primus International School is nearby / appears on the location map"],
      detail=["Says no distance is available for the school"],
      forbidden=["Gives a distance in km for any school"],
      reference="Primus International School appears on the location map but no distance is given",
      source="escalation"),
    Q("HG047", "hinglish", "rag", "I07", "medium", "RERA number share kar do",
      core=["The RERA number is PRM/KA/RERA/1251/446/PR/110522/004876"],
      detail=["The project is RERA registered"],
      forbidden=["Interprets what RERA registration guarantees"],
      reference="registered; PRM/KA/RERA/1251/446/PR/110522/004876"),
    Q("HG057", "hinglish", "rag", "I09", "easy", "home loan mein help mil jaayegi kya",
      variant="home loan me help mil jaayegi kia",
      core=["There is a dedicated home loan contact, Sastry", "Sastry's number is 8792582433"],
      forbidden=["Names approved banks / gives a list of approved banks"],
      reference="a dedicated home loan contact, Sastry on 8792582433"),
    Q("HG058", "hinglish", "escalation", "I09", "hard", "EMI kitni aayegi approx",
      core=["Says it cannot compute the EMI / does not have the payment details for it"],
      detail=["Offers to connect to sales"],
      forbidden=["Gives an EMI amount", "Gives a payment milestone or slab breakdown"],
      reference="payment schedule is not available to the agent; offers to connect to sales; no EMI "
                "computation - rate and tenure unknown", source="escalation"),
    Q("HG138", "hinglish", "rag", "I10", "medium", "ye location hi kyun choose karun",
      # Named as nearby is enough for "why this location"; a wrong km figure is still caught as a
      # contradiction against the reference.
      any_of=(["Close to Wipro Corporate office", "Close to RGA Tech Park", "Close to Decathlon",
               "Close to Manipal Hospital", "Close to Ecospace", "Close to Electronic City",
               "Close to Marathahalli"], 2),
      forbidden=["Gives a travel time in minutes", "Makes a claim about traffic",
                 "Makes a price-appreciation / investment-return claim"],
      reference="Wipro Corp 2.5 km, RGA Tech Park 2.2 km, Decathlon 2.5 km, Manipal Hospital 5.5 km, "
                "Ecospace 6.7 km, Electronic City 9 km, Marathahalli 9 km"),

    # ─────────────────────────────── Hindi (romanised)
    Q("HI003", "hi", "rag", "I01", "medium", "1239 wale flat ka total kitna padega sab milakar",
      variant="1239 wale flat ka total kitnaa padega sab milakar",
      core=["The 1,239 sft flat is approx Rs.1.54 crore all-in"],
      detail=["Says registration and GST are not included in that figure"],
      # The caller asked "sab milakar"; echoing that the figure covers EVERYTHING is false.
      forbidden=["States an exact registration or GST amount",
                 "Says the Rs.1.54 crore figure includes everything / all charges (it excludes GST, "
                 "registration, stamp duty, maintenance and corpus fund)"],
      reference="1239 sft all-in approx Rs.1.54 cr; must name what is excluded (registration, GST)"),
    Q("HI008", "hi", "rag", "I01", "medium", "1390 square feet ka kitna banta hai",
      variant="1390 square feet ka kitnaa banta hei",
      core=["The 1,390 sft flat is approx Rs.1.72 crore all-in"],
      detail=["Distinguishes the base flat cost from the all-in price",
              "The all-in price includes B/B car parking"],
      forbidden=["States an exact registration or GST amount",
                 "Explicitly calls Rs.1.72 crore the base / flat cost (it is the all-in price)"],
      reference="1390 sft approx Rs.1.72 cr all-in (B/B parking); base vs all-in must be "
                "distinguished"),
    Q("HI010", "hi", "rag", "I01", "hard", "saare charges milakar per square feet kitna padta hai",
      variant="saare charges milakar per square feet kitnaa padta hei",
      core=["There is an infrastructure charge of Rs.250 per sft on top of the base rate"],
      detail=["The Rs.250 per sft covers electrification, water, STP, WTP and power backup",
              "The charges are excluding GST"],
      forbidden=["States a combined / all-inclusive per-sft figure, i.e. any per-sft number other than Rs.11,000, Rs.11,499, Rs.250 or the listed extra charges (e.g. 11,749 or 11,250 per sq ft)",
                 "Says the Rs.250 per sft charge is refundable"],
      reference="Rs.250 per sft covering external electrification, water, STP, WTP and 100% power "
                "backup, excl GST; must not blend excl-GST items into one per-sft number"),
    Q("HI013", "hi", "rag", "I02", "medium", "ye corner flat ka charge kya hota hai",
      variant="ye corner flat ka charge kia hota hei",
      # Generalising is caught by the forbidden claim; naming the wings is detail.
      core=["The corner flat charge is Rs.50 per sft"],
      detail=["It applies only to specific corner units / wings"],
      forbidden=["Says that every flat / all flats pay the corner charge"],
      reference="Rs.50 per sft for corner flats; specific wings and unit numbers only"),
    Q("HI014", "hi", "rag", "I02", "easy", "clubhouse ka charge kitna hai",
      variant="clubhouse ka charge kitnaa hei",
      # Depends on configuration; naming the amount for any two of the three is a correct answer.
      any_of=(["2BHK clubhouse charge is Rs.2,50,000", "2.5BHK clubhouse charge is Rs.3,25,000",
               "3BHK clubhouse charge is Rs.3,25,000"], 2),
      forbidden=["Gives the clubhouse charge as a per-sft amount",
                 "Says the clubhouse charge is the same for every unit"],
      reference="KB cost sheets: clubhouse Rs.2,50,000 for a 2BHK, Rs.3,25,000 for a 2.5BHK, "
                "Rs.3,25,000 on all three 3BHK units", max_words=20,
      source="KB: 2BHK / 2.5BHK and 3BHK cost sheets"),
    Q("HI019", "hi", "rag", "I02", "easy", "parking price mein shamil hai kya",
      variant="parking price me shamil hei kia",
      core=["Yes, car parking is included in the published all-in price"],
      detail=["2BHK / 2.5BHK parking is Rs.3,00,000",
              "3BHK parking is Rs.5.5 lakh (covered) or Rs.6 lakh (two singles)"],
      forbidden=["Says parking is an extra cost on top of the published all-in price",
                 "Gives a price for a single (non-covered) parking space"],
      reference="KB 'Additional Charges': car parking is included in the published totals; where "
                "quoted separately, covered 5.5 lakh, two singles 6 lakh (3BHK); 2/2.5BHK sheet: "
                "car parking 3,00,000",
      source="KB: Additional Charges / cost sheets"),
    Q("HI020", "hi", "rag", "I02", "hard", "corner aur premium mein charge ka farak kya hai",
      variant="corner aur premium me charge ka farak kia hei",
      core=["Corner flats are charged Rs.50 per sft", "Premium flats are charged Rs.75 per sft"],
      detail=["Flats that are both corner and premium are charged Rs.100 per sft",
              "These charges apply only to specific units / wings"],
      forbidden=["Says these charges apply to every unit"],
      reference="KB 'Additional Charges': corner Rs.50/sft, premium Rs.75/sft (2BHK cost sheet), "
                "corner-and-premium Rs.100/sft; specific units only",
      source="KB: Additional Charges Over the Cost Sheet"),
    Q("HI025", "hi", "rag", "I03", "easy", "ek BHK hai kya aapke paas",
      variant="ek BHK hei kia aapke paas",
      core=["Yes, 1BHK units are available", "1BHK units are in Tower D only"],
      detail=["1BHK units are 619 to 646 sft", "There are also 1.5BHK units of 780 sft"],
      forbidden=["Says 1BHK units are in Tower A, B or C"],
      reference="1BHK 619 to 646 sft; 1.5BHK 780 sft; Tower D only"),
    Q("HI029", "hi", "rag", "I04", "easy", "khel kood ki kya suvidha hai",
      variant="khel kood ki kia suvidha hei",
      # "Sports facilities" is not only outdoor: the clubhouse squash / badminton courts and the
      # pool are real sports facilities too (brochure), so naming them is a correct answer.
      any_of=(["Tennis court", "Basketball court", "Mini soccer", "Cricket practice nets",
               "Skating rink", "Rock climbing wall", "Trampoline park",
               "Squash court (clubhouse)", "Badminton court (clubhouse)", "Swimming pool"], 3),
      forbidden=["Presents squash as an outdoor facility"],
      reference="outdoor: tennis court, basketball court, mini soccer, cricket practice nets, "
                "skating rink, rock climbing wall, trampoline park; indoor in the clubhouse: squash, "
                "badminton (name 4 max)"),
    Q("HI036", "hi", "rag", "I05", "medium", "khidkiyan UPVC hain kya",
      variant="khidkiyan UPVC hain kia",
      core=["Yes, the windows are UPVC sliding windows"],
      detail=["The windows have glass", "The windows have mosquito mesh",
              "The windows have safety grills", "French doors do not have a safety grill"],
      forbidden=["Says the windows are aluminium", "Says the French doors have safety grills"],
      reference="UPVC sliding windows with glass, mosquito mesh and safety grills; French doors have "
                "no grill"),
    Q("HI040", "hi", "rag", "I06", "easy", "project theek theek kahan par hai",
      variant="project thik thik kahan par hei",
      core=["The project is at Chikkanayakanahalli, off Sarjapur Road"],
      detail=["It is in Bengaluru"],
      forbidden=["Gives a pin code", "Gives a survey number"],
      reference="Chikkanayakanahalli, off Sarjapur Road, Bengaluru", max_words=20),
    Q("HI041", "hi", "rag", "I06", "easy", "Wipro se kitni door hai",
      variant="Wipro se kitni door hei",
      core=["Wipro Corporate office is 2.5 km away"],
      forbidden=["Gives a travel time in minutes", "Makes a claim about traffic"],
      reference="Wipro Corp 2.5 km"),
    Q("HI044", "hi", "rag", "I06", "medium", "wahan metro aane wala hai kya",
      variant="wahan metro aane wala hei kia",
      core=["Says it does not have metro information for the project"],
      forbidden=["Names a metro line", "Names a metro station", "Gives a metro timeline",
                 "Says a metro is coming"],
      reference="no metro information is in the project material", source="escalation"),
    Q("HI046", "hi", "rag", "I07", "medium", "RERA approved hai kya",
      variant="RERA approved hei kia",
      core=["Yes, the project is RERA registered"],
      detail=["The RERA number is PRM/KA/RERA/1251/446/PR/110522/004876"],
      forbidden=["Interprets what RERA registration guarantees"],
      reference="registered; PRM/KA/RERA/1251/446/PR/110522/004876"),
    Q("HI049", "hi", "escalation", "I07", "medium", "OC mil gaya hai kya",
      variant="OC mil gaya hei kia",
      core=["Says it does not have the OC / CC status"],
      detail=["Offers to connect to sales"],
      forbidden=["Says the OC has been received", "Says the OC is pending"],
      reference="occupancy and completion certificate status is not available to the agent",
      source="escalation"),
    Q("HI052", "hi", "escalation", "I08", "hard",
      "aapke sales wale ne agle saal ka bola tha, confirm kar dijiye",
      # Not confirming the caller's date is enforced by the forbidden list; the core fact is the
      # one positive behaviour (a bundled "doesn't confirm AND says..." fact graded unstably).
      core=["Says it does not have the possession timing"],
      detail=["Offers to connect to sales"],
      forbidden=["Confirms the possession date the caller mentioned",
                 "Gives any possession date, month, quarter or year", "Says possession is 'soon'"],
      reference="agent does not have possession timing; offers to connect to sales; must not "
                "confirm a date fed by the caller", source="escalation"),
    Q("HI059", "hi", "escalation", "I09", "easy", "registration ka kharcha kitna aayega",
      variant="registration ka kharcha kitnaa aayega",
      core=["Registration and stamp duty are charged at actuals"],
      forbidden=["Gives a rupee figure for registration or stamp duty",
                 "Gives a percentage for registration or stamp duty"],
      reference="registration and stamp duty are at actuals; no figure", source="escalation"),
]

assert len(GOLDEN) == 50 and len({q["id"] for q in GOLDEN}) == 50
assert all(q["core"] or q["any_of"] for q in GOLDEN), "every question needs a pass condition"
