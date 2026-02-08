def print_step(title: str, width: int = 75) -> None:
    line = "=" * width
    print(f"\n{line}")
    print(f"{title.center(width)}")
    print(f"{line}")
