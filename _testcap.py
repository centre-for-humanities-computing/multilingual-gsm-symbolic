from multilingual_gsm_symbolic._helpers import capitalize_sentences as cs

tests = [
    # abbreviation followed by lowercase letter: case is PRESERVED (matches stored source)
    ("day starts 8:00 a.m. then joins", "Day starts 8:00 a.m. then joins"),
    ("mckenna starts at 8:00 a.m. she works later", "Mckenna starts at 8:00 a.m. she works later"),
    ("angie bought 3 lbs. of coffee", "Angie bought 3 lbs. of coffee"),
    ("use e.g. apples and pears", "Use e.g. apples and pears"),
    ("weighs 2 kg. next, cut it", "Weighs 2 kg. next, cut it"),
    # normal sentences still capitalize
    ("he left. then she stayed. it rained", "He left. Then she stayed. It rained"),
    ("it costs $5. and tax is $1", "It costs $5. And tax is $1"),
    # words merely ending in abbreviation letters are unaffected
    ("the program will end. then dinner", "The program will end. Then dinner"),
    ("i sat in the left. seat", "I sat in the left. Seat"),
]
ok = True
for inp, want in tests:
    got = cs(inp)
    good = got == want
    ok = ok and good
    if not good:
        print("FAIL", repr(inp), "->", repr(got), "want", repr(want))
print("ALL PASS" if ok else "FAILURES")
