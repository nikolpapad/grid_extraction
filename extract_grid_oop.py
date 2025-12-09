from oop import GridExtractor

extractor = GridExtractor("C:/Users/nikol/Downloads/page_22a.png")

extractor.load_image()
extractor.preprocess()
extractor.detect_lines()
extractor.extract_grid_lines()
extractor.extract_cells()

instructions = extractor.build_instructions(start_corner="bottom-left", alternate=False)

print("\n--- Pattern Instructions ---")
for line in instructions:
    print(line)

extractor.show_results()
