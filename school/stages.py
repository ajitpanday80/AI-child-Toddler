"""The school: a ladder of stages, from values up to adulthood.

Every stage teaches ALL subjects at that level (a child is not an expert in one thing):
  values    moral stories, ethics, virtues          language  reading & English
  math      numbers to algebra                      science   nature, physics, chemistry, biology
  history   people and the past                     civics    rules, government, political science
  mind      psychology & how people think           heart     recognising feelings (labelled examples)
  judgment  is this action right or wrong? (labelled examples)
  extra     anything YOU drop into content/<stage>/ (PDFs, text, web links)

All material is fetched automatically. To change what a stage teaches, edit the lists below.
To go further than Grade 8, add Stage(...) entries at the end (see README).
"""
from dataclasses import dataclass, field
from typing import Dict, List

from .data import Source

SIMPLE, EN = "simple", "en"


def wiki(subject, titles, lang=SIMPLE):
    return Source("wiki", (lang, titles.split("|")), subject)


@dataclass
class Stage:
    name: str
    folder: str                         # your own material goes in content/<folder>/
    skill: str
    sources: List[Source]
    cloze_pass: float = 0.55            # average on "pick the missing word" across subjects (chance 25%)
    tasks_pass: Dict[str, float] = field(default_factory=dict)   # moral / emotion / judgment marks (labelled exams)
    review_tol: float = 0.10            # earlier stages may not drop more than this below the mark they passed with


MCG = {1: 14640, 2: 14668, 3: 14766, 4: 14880, 5: 15040, 6: 16751}   # McGuffey's graded readers (Project Gutenberg)

STAGES = [
    Stage("Values", "values", "moral stories, ethics, kindness, right & wrong", [
        Source("gutenberg", 19994, fables=True), Source("gutenberg", 18442, subject="values"), Source("ethics", 8000),
        wiki("values", "Honesty|Kindness|Fairness|Courage|Friendship|Respect|Responsibility|Forgiveness|Patience|Generosity|Truth|Bullying|Ethics|Morality|Golden Rule|Empathy|Compassion|Gratitude|Justice|Peace|Sharing|Cheating|Stealing|Lie"),
        wiki("mind", "Emotion|Happiness|Sadness|Anger|Fear|Love|Surprise|Pain|Hope"),
    ], cloze_pass=0.42, tasks_pass={"judgment": 0.54}),   # "moral" (fable -> true moral) is shown but not required: too little data to learn it

    Stage("Pre-Nursery", "pre-nursery", "rhymes, colours, shapes, family, feelings", [
        Source("gutenberg", 10607, verse=True), Source("gutenberg", 39784, verse=True),
        wiki("math", "Number|Zero|Shape|Circle|Square|Triangle|Counting|Size|Money"),
        wiki("science", "Color|Animal|Dog|Cat|Bird|Fish|Water|Sun|Moon|Rain|Plant|Tree|Fruit|Milk|Day|Night"),
        wiki("civics", "Family|Mother|Father|Home|School|Teacher|Friend|Doctor|Police"),
        wiki("mind", "Emotion|Smile|Hug|Sleep|Dream|Laughter|Crying"),
    ], cloze_pass=0.45),

    Stage("Nursery", "nursery", "simple stories, counting, animals, the world around me", [
        Source("tinystories", (0, 1_200_000)),
        wiki("math", "Addition|Subtraction|Counting|Time|Clock|Calendar|Week|Month|Year|Length"),
        wiki("science", "Weather|Season|Spring|Summer|Autumn|Winter|Air|Wind|Cloud|Snow|Flower|Insect|Farm|Forest|Sea|Mountain"),
        wiki("history", "History|Past|Holiday|Birthday|Toy|Game|Song|Music"),
        wiki("civics", "Community|Village|Town|City|Farmer|Fireman|Nurse|Shop|Rule"),
        wiki("mind", "Joy|Sadness|Anger|Fear|Love|Friendship|Sharing|Sleep|Memory"),
    ], cloze_pass=0.50),

    Stage("Grade 1", "grade1", "reading, +/-, living things, rules, feelings", [
        Source("gutenberg", MCG[1]),
        Source("emotion", 3000),
        wiki("math", "Addition|Subtraction|Even and odd numbers|Number line|Coin|Fraction|Measurement|Weight|Temperature|Dozen"),
        wiki("science", "Magnet|Light|Sound|Heat|Solid|Liquid|Gas|Earth|Sun|Moon|Star|Planet|Mammal|Reptile|Amphibian|Skeleton|Human body"),
        wiki("history", "Calendar|Flag|National holiday|Christmas|Eid al-Fitr|Diwali|Thanksgiving|Museum|Library"),
        wiki("civics", "Law|Government|Country|Mayor|Vote|Citizen|Flag|Rule|Right|Police|Community"),
        wiki("mind", "Emotion|Memory|Dream|Sleep|Anger|Fear|Shyness|Jealousy|Pride|Embarrassment|Boredom"),
        wiki("values", "Honesty|Kindness|Fairness|Courage|Sharing|Gratitude|Respect|Obedience|Bullying"),
    ], cloze_pass=0.50, tasks_pass={"emotion": 0.35}),

    Stage("Grade 2", "grade2", "stories, multiplication, water & weather, community", [
        Source("gutenberg", MCG[2]), Source("emotion", 3500),
        wiki("math", "Multiplication|Division|Place value|Odd number|Even number|Geometry|Line|Angle|Perimeter|Area|Time|Money"),
        wiki("science", "Water cycle|Weather|Rain|Thunderstorm|Rainbow|Volcano|Earthquake|Soil|Rock|Fossil|Plant|Seed|Photosynthesis|Pollination|Food chain|Habitat"),
        wiki("history", "Ancient Egypt|Pyramid|Stone Age|Dinosaur|Explorer|Christopher Columbus|Wheel|Writing|Alphabet|Paper"),
        wiki("civics", "Election|Democracy|Constitution|President|Prime minister|Parliament|Court|Judge|Taxes|Public library|Fire department"),
        wiki("mind", "Feeling|Friendship|Empathy|Kindness|Teamwork|Anger management|Stress|Happiness|Courage"),
        wiki("values", "Responsibility|Honesty|Forgiveness|Patience|Generosity|Fairness|Cheating|Stealing|Respect"),
    ], cloze_pass=0.50, tasks_pass={"emotion": 0.40}),

    Stage("Grade 3", "grade3", "chapter reading, fractions, forces, ancient history, rights", [
        Source("gutenberg", MCG[3]), Source("emotion", 4000),
        wiki("math", "Fraction|Decimal|Multiplication|Division|Perimeter|Area|Triangle|Rectangle|Polygon|Symmetry|Graph|Probability"),
        wiki("science", "Force|Gravity|Friction|Energy|Motion|Simple machine|Lever|Electricity|Magnet|Light|Sound|Atom|Ecosystem|Food chain|Adaptation"),
        wiki("history", "Ancient Greece|Ancient Rome|Indus Valley Civilisation|Mesopotamia|Ancient China|Silk Road|Middle Ages|Renaissance|Mahatma Gandhi|Abraham Lincoln"),
        wiki("civics", "Human rights|Children's rights|Equality|Freedom|Citizenship|Tax|Court|Constitution|Democracy|Government"),
        wiki("mind", "Empathy|Emotion|Bullying|Self-esteem|Motivation|Curiosity|Memory|Learning|Personality"),
        wiki("values", "Ethics|Morality|Justice|Integrity|Honesty|Compassion|Human rights|Tolerance|Conscience"),
    ], cloze_pass=0.50, tasks_pass={"emotion": 0.40}),

    Stage("Grade 4", "grade4", "longer reading, geometry, matter, civilisations, right vs wrong", [
        Source("gutenberg", MCG[4]), Source("emotion", 4500), Source("ethics", 3500),
        wiki("math", "Fraction|Decimal|Percentage|Ratio|Prime number|Factor (arithmetic)|Angle|Circle|Volume|Geometry|Statistics|Average"),
        wiki("science", "Matter|Atom|Element|Chemical reaction|Mixture|Solar System|Energy|Electricity|Circuit|Cell (biology)|Digestion|Respiration|Climate"),
        wiki("history", "Roman Empire|Byzantine Empire|Maya civilization|Aztec|Vikings|Mongol Empire|Mughal Empire|Silk Road|Printing press|Age of Discovery"),
        wiki("civics", "Government|Separation of powers|Federalism|Democracy|Monarchy|Republic|Constitution|Legislature|Law|United Nations"),
        wiki("mind", "Emotion|Empathy|Anxiety|Self-control|Peer pressure|Bullying|Friendship|Conflict|Cognitive development"),
        wiki("values", "Ethics|Honesty|Responsibility|Fairness|Justice|Courage|Integrity|Respect|Tolerance|Golden Rule"),
    ], cloze_pass=0.50, tasks_pass={"emotion": 0.40, "judgment": 0.55}),

    Stage("Grade 5", "grade5", "classic reading, algebra start, physics & life science, world history, ethics", [
        Source("gutenberg", MCG[5]), Source("emotion", 5500), Source("ethics", 5500),
        wiki("math", "Algebra|Equation|Variable (mathematics)|Exponentiation|Square root|Integer|Negative number|Coordinate system|Probability|Pythagorean theorem|Pi|Geometry"),
        wiki("science", "Newton's laws of motion|Velocity|Acceleration|Work (physics)|Heat|Temperature|Sound|Light|Electromagnetism|Photosynthesis|Evolution|DNA|Chemical element|Periodic table"),
        wiki("history", "Industrial Revolution|American Revolution|French Revolution|Slavery|Abolitionism|Colonialism|Indian independence movement|World War I|World War II|Cold War"),
        wiki("civics", "Democracy|Dictatorship|Political party|Election|Freedom of speech|Human rights|Constitution|Rule of law|Corruption|Propaganda|Citizenship"),
        wiki("mind", "Psychology|Emotion|Empathy|Motivation|Memory|Learning|Personality|Intelligence|Stress|Emotional intelligence"),
        wiki("values", "Ethics|Morality|Utilitarianism|Golden Rule|Justice|Human rights|Honesty|Integrity|Conscience|Virtue"),
    ], cloze_pass=0.50, tasks_pass={"emotion": 0.40, "judgment": 0.58}),

    Stage("Grade 6", "grade6", "advanced reading, ratio & geometry, chemistry, history of ideas, moral reasoning", [
        Source("gutenberg", MCG[6]), Source("emotion", 6500), Source("ethics", 7000),
        wiki("math", "Ratio|Proportion|Linear equation|Function (mathematics)|Inequality (mathematics)|Statistics|Mean|Median|Geometry|Trigonometry|Prime number|Logic", EN),
        wiki("science", "Chemical bond|Acid|Base (chemistry)|Chemical reaction|Newton's laws of motion|Kinetic energy|Potential energy|Electric current|Cell (biology)|Genetics|Natural selection|Climate change", EN),
        wiki("history", "Scientific Revolution|Enlightenment|Industrial Revolution|Renaissance|Reformation|Age of Discovery|Atlantic slave trade|Women's suffrage|Civil rights movement", EN),
        wiki("civics", "Political science|Democracy|Authoritarianism|Social contract|Separation of powers|Rule of law|Universal Declaration of Human Rights|Political philosophy|Liberty|Equality", EN),
        wiki("mind", "Developmental psychology|Piaget's theory of cognitive development|Empathy|Emotion|Cognitive bias|Peer pressure|Motivation|Self-esteem|Attachment theory|Theory of mind", EN),
        wiki("values", "Ethics|Moral development|Kohlberg's stages of moral development|Utilitarianism|Deontology|Virtue ethics|Social justice|Altruism|Moral reasoning|Conscience", EN),
    ], cloze_pass=0.50, tasks_pass={"emotion": 0.40, "judgment": 0.58}),
]
