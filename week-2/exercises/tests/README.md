# Week 2 Exercise Tests

Run all tests from the week-2 directory:

```bash
cd week-2
python -m pytest exercises/tests/ -v
```

Run specific exercise tests:

```bash
# Exercise 1 (NumPy)
python -m pytest exercises/tests/test_exercise_1.py -v

# Exercise 2 (Pandas)
python -m pytest exercises/tests/test_exercise_2.py -v
```

Run with coverage:

```bash
python -m pytest exercises/tests/ -v --cov=exercises
```
