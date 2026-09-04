from __future__ import annotations

import importlib
import inspect
import pathlib
import sys
import time
import traceback


def main() -> int:
    test_dir = pathlib.Path(__file__).parent / "tests"
    test_files = sorted(test_dir.glob("test_*.py"))

    total = 0
    passed = 0
    failed = 0
    failures = []

    print("=" * 70)
    print("🚀 Running SIH26055 RF Environment V2 Test Suite")
    print("=" * 70)

    start_time = time.time()

    for f in test_files:
        mod_name = f"tests.{f.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"\n❌ Failed to import test module {mod_name}: {e}")
            traceback.print_exc()
            failed += 1
            continue

        test_funcs = [
            (name, obj)
            for name, obj in inspect.getmembers(mod)
            if name.startswith("test_") and inspect.isfunction(obj)
        ]

        if not test_funcs:
            continue

        print(f"\n📁 {f.name} ({len(test_funcs)} tests):")

        for name, func in test_funcs:
            total += 1
            t0 = time.time()
            try:
                func()
                elapsed = (time.time() - t0) * 1000
                print(f"   ✅ {name:<50} ({elapsed:.1f} ms)")
                passed += 1
            except Exception as e:
                elapsed = (time.time() - t0) * 1000
                print(f"   ❌ {name:<50} FAILED ({elapsed:.1f} ms)")
                failed += 1
                failures.append((mod_name, name, e, traceback.format_exc()))

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed} PASSED, {failed} FAILED in {total_time:.2f}s")
    print("=" * 70)

    if failures:
        print("\n💥 Failure Details:")
        for mod, name, exc, tb in failures:
            print(f"\n--- {mod}.{name} ---")
            print(tb)
        return 1

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
