#!/usr/bin/env python3
"""Homework 05: compare Boyer–Moore, KMP, and Rabin–Karp substring search.

Uses the original course articles, included as UTF-8 text files under articles/.
Requires Python >= 3.10 and no third-party packages.

Examples:
    python3 main.py
    python3 main.py --self-test
    python3 main.py --article-1 /path/article1.txt --article-2 /path/article2.txt
    python3 main.py --number 20 --repeat 7

The program prints a Markdown report and does not create extra files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import random
from statistics import mean, median
import sys
from timeit import repeat as timeit_repeat
from typing import Callable


SearchFunction = Callable[[str, str], int]
ARTICLE_DIR = Path(__file__).resolve().parent / "articles"


def _good_suffix_shifts(pattern: str) -> list[int]:
    """Precompute strong good-suffix shifts for Boyer–Moore."""
    length = len(pattern)
    shifts = [0] * (length + 1)
    borders = [0] * (length + 1)
    i, j = length, length + 1
    borders[i] = j

    while i > 0:
        while j <= length and pattern[i - 1] != pattern[j - 1]:
            if shifts[j] == 0:
                shifts[j] = j - i
            j = borders[j]
        i -= 1
        j -= 1
        borders[i] = j

    j = borders[0]
    for i in range(length + 1):
        if shifts[i] == 0:
            shifts[i] = j
        if i == j:
            j = borders[j]
    return shifts


def boyer_moore(text: str, pattern: str) -> int:
    """Return first occurrence, using bad-character and good-suffix heuristics."""
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    if m > n:
        return -1

    last_seen = {char: i for i, char in enumerate(pattern)}
    suffix_shift = _good_suffix_shifts(pattern)
    start = 0

    while start <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[start + j]:
            j -= 1
        if j < 0:
            return start

        bad_character_shift = j - last_seen.get(text[start + j], -1)
        start += max(1, bad_character_shift, suffix_shift[j + 1])
    return -1


def knuth_morris_pratt(text: str, pattern: str) -> int:
    """Return first occurrence; worst-case O(n + m) time."""
    if not pattern:
        return 0
    m = len(pattern)
    lps = [0] * m
    prefix_length = 0

    for i in range(1, m):
        while prefix_length > 0 and pattern[i] != pattern[prefix_length]:
            prefix_length = lps[prefix_length - 1]
        if pattern[i] == pattern[prefix_length]:
            prefix_length += 1
            lps[i] = prefix_length

    matched = 0
    for i, char in enumerate(text):
        while matched > 0 and char != pattern[matched]:
            matched = lps[matched - 1]
        if char == pattern[matched]:
            matched += 1
        if matched == m:
            return i - m + 1
    return -1


def rabin_karp(text: str, pattern: str) -> int:
    """Return first occurrence; rolling hash with exact collision verification."""
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    if m > n:
        return -1

    base, modulus = 256, 1_000_000_007
    leading_power = pow(base, m - 1, modulus)
    target_hash = window_hash = 0

    for i in range(m):
        target_hash = (target_hash * base + ord(pattern[i])) % modulus
        window_hash = (window_hash * base + ord(text[i])) % modulus

    for start in range(n - m + 1):
        if target_hash == window_hash and text[start : start + m] == pattern:
            return start
        if start < n - m:
            window_hash = (
                base * (window_hash - ord(text[start]) * leading_power)
                + ord(text[start + m])
            ) % modulus
    return -1


ALGORITHMS: dict[str, SearchFunction] = {
    "Boyer–Moore": boyer_moore,
    "Knuth–Morris–Pratt": knuth_morris_pratt,
    "Rabin–Karp": rabin_karp,
}


def read_article(path: Path) -> str:
    """Read a course article: UTF-8 or original Windows-1251 encoding."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            article = raw.decode(encoding)
            if not article.strip():
                raise ValueError(f"Стаття порожня: {path}")
            return article
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Непідтримуване кодування: {path}")


def choose_patterns(text: str, size: int = 28) -> tuple[str, str]:
    """Select a real late-occurring substring and an equally long absent one."""
    if len(text) < size:
        raise ValueError("Стаття занадто коротка для вибору підрядка")

    # Make the positive case nontrivial: aim to find the first match late.
    beginning = len(text) // 2
    present = None
    for start in range(beginning, len(text) - size + 1):
        substring = text[start : start + size]
        if (
            substring.isprintable()
            and substring.count(" ") >= 2
            and text.find(substring) == start
        ):
            present = substring
            break
    if present is None:
        raise ValueError("Не знайдено підходящого наявного підрядка")

    # The sentinel does not occur in the article, so absence is guaranteed.
    sentinel = next(
        (char for char in ("⟐", "☃", "¤", "※") if char not in text), None
    )
    if sentinel is None:
        raise ValueError("Не вдалося створити гарантовано відсутній підрядок")
    absent = present[:-1] + sentinel
    assert absent not in text
    return present, absent


def measure(text: str, pattern: str, number: int, repeats: int) -> dict[str, float]:
    """Measure median milliseconds per search, including pattern preparation."""
    expected = text.find(pattern)
    results: dict[str, float] = {}
    for name, search in ALGORITHMS.items():
        result = search(text, pattern)
        if result != expected:
            raise AssertionError(
                f"{name}: очікувалося {expected}, отримано {result}"
            )
        measurements = timeit_repeat(
            lambda search=search: search(text, pattern),
            number=number,
            repeat=repeats,
        )
        results[name] = median(measurements) * 1000.0 / number
    return results


def self_test() -> None:
    """Run deterministic edge and randomized correctness tests."""
    cases = [
        ("", ""), ("", "x"), ("abc", ""), ("abc", "abcd"),
        ("abc", "abc"), ("abcabc", "bc"), ("aaaaaaa", "aaa"),
        ("aaaaab", "aaab"), ("ababababac", "ababac"),
        ("abcababcab", "abcab"), ("абракадабра", "кад"),
        ("🐍Python🐍", "🐍"), ("a" * 2000 + "b", "a" * 400 + "b"),
        ("a" * 2000 + "b", "a" * 400 + "c"),
    ]
    rng = random.Random(2026)
    alphabet = "abcabáїїЄ😀123"
    for _ in range(600):
        text = "".join(rng.choices(alphabet, k=rng.randrange(0, 110)))
        pattern = "".join(rng.choices(alphabet, k=rng.randrange(0, 25)))
        cases.append((text, pattern))

    checks = 0
    for text, pattern in cases:
        expected = text.find(pattern)
        for name, algorithm in ALGORITHMS.items():
            result = algorithm(text, pattern)
            if result != expected:
                raise AssertionError(
                    f"{name}: {text!r}, {pattern!r}: {result} != {expected}"
                )
            checks += 1
    print(f"PASS: {len(cases)} прикладів, {checks} перевірок алгоритмів")


def report(articles: tuple[str, str], number: int, repeats: int) -> None:
    """Print self-contained Markdown report to standard output."""
    measurements: list[tuple[int, str, dict[str, float]]] = []
    print("# Порівняння алгоритмів пошуку підрядка\n")
    print("## Методика\n")
    print(
        f"- Середовище: Python {platform.python_version()}, "
        f"{platform.system()} {platform.machine()}."
    )
    print(f"- `timeit.repeat`: `number={number}`, `repeat={repeats}`.")
    print("- Метрика: медіана повторів, мілісекунди за один пошук.")
    print("- Час завантаження файлів не враховано; підготовка шаблону входить.")
    print("- Перевірка результатів: порівняння індексів із `str.find()`.\n")
    print("## Результати\n")
    print("| Стаття | Тип підрядка | Boyer–Moore, мс | KMP, мс | Rabin–Karp, мс | Найшвидший |")
    print("|---|---|---:|---:|---:|---|")

    for number_of_article, article in enumerate(articles, start=1):
        present, absent = choose_patterns(article)
        for label, pattern in (("наявний", present), ("відсутній", absent)):
            times = measure(article, pattern, number, repeats)
            winner = min(times, key=times.get)
            measurements.append((number_of_article, label, times))
            print(
                f"| {number_of_article} | {label} | "
                f"{times['Boyer–Moore']:.5f} | "
                f"{times['Knuth–Morris–Pratt']:.5f} | "
                f"{times['Rabin–Karp']:.5f} | **{winner}** |"
            )

    print("\n## Підрядки\n")
    for index, article in enumerate(articles, start=1):
        present, absent = choose_patterns(article)
        print(f"- Стаття {index} ({len(article)} символів): ")
        print(f"  - наявний: `{present}` (позиція {article.find(present)});")
        print(f"  - відсутній: `{absent}`.")

    print("\n## Висновки\n")
    for index in (1, 2):
        selected = [times for article_num, _, times in measurements if article_num == index]
        averages = {
            name: mean(sample[name] for sample in selected) for name in ALGORITHMS
        }
        winner = min(averages, key=averages.get)
        print(
            f"- **Стаття {index}:** найшвидший за середнім двох сценаріїв "
            f"**{winner}** ({averages[winner]:.5f} мс)."
        )

    averages = {
        name: mean(result[name] for _, _, result in measurements)
        for name in ALGORITHMS
    }
    winner = min(averages, key=averages.get)
    print(
        f"- **Загалом:** найшвидший за середнім чотирьох сценаріїв "
        f"**{winner}** ({averages[winner]:.5f} мс)."
    )
    print(
        "- **Пояснення:** Boyer–Moore використовує евристики поганого символу й "
        "хорошого суфікса; KMP має гарантовану складність O(n + m); "
        "Rabin–Karp використовує ковзний хеш і перевірку збігів."
    )
    print(
        "- **Обмеження:** час залежить від тексту, шаблону, реалізації та "
        "середовища. Ці результати не доводять універсальну перевагу "
        "одного алгоритму."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--article-1", type=Path, default=ARTICLE_DIR / "article_1.txt"
    )
    parser.add_argument(
        "--article-2", type=Path, default=ARTICLE_DIR / "article_2.txt"
    )
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.number <= 0 or args.repeat <= 0:
        parser.error("--number і --repeat мають бути додатними цілими числами")

    try:
        articles = read_article(args.article_1), read_article(args.article_2)
        report(articles, args.number, args.repeat)
    except (OSError, ValueError, AssertionError) as exc:
        print(f"Помилка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
