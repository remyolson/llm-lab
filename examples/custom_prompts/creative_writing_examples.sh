#!/bin/bash
# Creative Writing Examples
# Demonstrates different genres, styles, and creative formats

echo "=== Creative Writing Examples ==="
echo "Testing creative writing across different genres and styles"
echo

# Example 1: Science Fiction Short Story
echo "1. Testing Science Fiction Story..."
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing/story_generation.txt \
  --prompt-variables '{
    "genre": "science fiction",
    "word_count": "600-800",
    "protagonist": "Dr. Maya Chen, a quantum physicist",
    "setting": "Research station on Europa, Jupiter'\''s moon, in 2157",
    "conflict": "Discovery of alien technology that challenges human understanding of physics",
    "theme": "The responsibility that comes with revolutionary scientific discovery",
    "tone": "cerebral and suspenseful",
    "pov": "third person limited",
    "target_audience": "adult science fiction readers"
  }' \
  --models gpt-4,claude-3-opus \
  --parallel \
  --output-format markdown \
  --output-dir ./results/examples/creative-writing/sci-fi

echo "✅ Science fiction story completed"
echo

# Example 2: Mystery Short Story
echo "2. Testing Mystery Story..."
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing/story_generation.txt \
  --prompt-variables '{
    "genre": "mystery",
    "word_count": "500-700",
    "protagonist": "Detective Sarah Walsh, 15-year veteran",
    "setting": "Small coastal town during a winter storm",
    "conflict": "Locked-room murder at the local lighthouse",
    "theme": "Truth hidden beneath layers of small-town secrets",
    "tone": "atmospheric and suspenseful",
    "pov": "third person limited"
  }' \
  --models gpt-4,claude-3-sonnet \
  --output-format json,markdown \
  --output-dir ./results/examples/creative-writing/mystery

echo "✅ Mystery story completed"
echo

# Example 3: Poetry Generation
echo "3. Testing Poetry Generation..."
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing/creative_writing.txt \
  --prompt-variables '{
    "writer_type": "contemporary poet",
    "content_type": "poem",
    "genre": "free verse",
    "writing_style": "lyrical and introspective",
    "target_length": "20-30 lines",
    "theme": "urban isolation and unexpected human connections",
    "setting": "city subway system at rush hour",
    "mood": "melancholic yet hopeful",
    "style_notes": "Use concrete imagery, enjambment, and subtle metaphors. Avoid forced rhymes."
  }' \
  --models gpt-4,claude-3-sonnet,gemini-pro \
  --parallel \
  --metrics creativity,coherence \
  --output-dir ./results/examples/creative-writing/poetry

echo "✅ Poetry generation completed"
echo

# Example 4: Screenplay Dialogue
echo "4. Testing Screenplay Dialogue..."
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing/creative_writing.txt \
  --prompt-variables '{
    "writer_type": "screenwriter",
    "content_type": "dialogue",
    "genre": "psychological thriller",
    "writing_style": "tense with heavy subtext",
    "target_length": "400-500 words",
    "characters": "ALEX (30s, tech entrepreneur hiding secrets), DR. MORGAN (50s, FBI behavioral analyst), JAMIE (20s, Alex'\''s assistant who knows too much)",
    "setting": "FBI interrogation room, late evening",
    "mood": "claustrophobic and increasingly tense",
    "constraints": "Each character must have distinct speech patterns. Build tension through what isn'\''t said. Include minimal but effective stage directions.",
    "theme": "the price of success built on deception"
  }' \
  --models gpt-4,claude-3-sonnet \
  --output-format json,markdown \
  --output-dir ./results/examples/creative-writing/dialogue

echo "✅ Dialogue generation completed"
echo

# Example 5: Fantasy World-Building
echo "5. Testing Fantasy Story..."
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing/story_generation.txt \
  --prompt-variables '{
    "genre": "fantasy",
    "word_count": "700-900",
    "protagonist": "Kira, a reluctant mage who fears her own power",
    "setting": "The Floating Cities of Aethermoor, connected by bridges of crystallized air",
    "conflict": "Ancient magic is failing, causing cities to fall from the sky",
    "theme": "accepting responsibility for gifts we didn'\''t choose",
    "tone": "epic yet intimate",
    "pov": "third person limited",
    "target_audience": "young adult fantasy readers"
  }' \
  --models gpt-4,claude-3-opus \
  --output-dir ./results/examples/creative-writing/fantasy

echo "✅ Fantasy story completed"
echo

# Example 6: Batch Testing Different Genres
echo "6. Testing Multiple Genres..."
declare -a genres=(
  "horror:atmospheric and unsettling:A cursed antique music box"
  "romance:emotional and character-driven:Two rival food truck owners"
  "historical fiction:authentic and immersive:A telegraph operator during the 1906 San Francisco earthquake"
  "western:gritty and authentic:A female bounty hunter in 1880s Arizona"
)

for genre_info in "${genres[@]}"; do
  IFS=':' read -r genre style premise <<< "$genre_info"
  echo "   Testing genre: $genre"

  python scripts/run_benchmarks.py \
    --prompt-file templates/creative_writing/story_generation.txt \
    --prompt-variables "{
      \"genre\": \"$genre\",
      \"word_count\": \"400-600\",
      \"protagonist\": \"$premise\",
      \"writing_style\": \"$style\",
      \"theme\": \"overcoming adversity through inner strength\"
    }" \
    --models gpt-4o-mini,claude-3-haiku \
    --limit 1 \
    --output-dir "./results/examples/creative-writing/genres/$genre"
done

echo "✅ Genre batch testing completed"
echo

# Example 7: Style Comparison
echo "7. Testing Writing Style Variations..."
python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing/story_generation.txt \
  --prompt-variables '{
    "genre": "contemporary fiction",
    "word_count": "400-500",
    "protagonist": "Emma, a librarian discovering a hidden room",
    "writing_style": "minimalist and sparse",
    "theme": "finding magic in ordinary places",
    "tone": "quiet and contemplative"
  }' \
  --models gpt-4 \
  --limit 2 \
  --output-dir ./results/examples/creative-writing/style-test/minimalist

python scripts/run_benchmarks.py \
  --prompt-file templates/creative_writing/story_generation.txt \
  --prompt-variables '{
    "genre": "contemporary fiction",
    "word_count": "400-500",
    "protagonist": "Emma, a librarian discovering a hidden room",
    "writing_style": "rich and descriptive",
    "theme": "finding magic in ordinary places",
    "tone": "lush and atmospheric"
  }' \
  --models gpt-4 \
  --limit 2 \
  --output-dir ./results/examples/creative-writing/style-test/descriptive

echo "✅ Style comparison completed"
echo

echo "=== All Creative Writing Examples Completed ==="
echo "Results saved to: ./results/examples/creative-writing/"
echo "View results:"
echo "  - Sci-fi: cat results/examples/creative-writing/sci-fi/*.md"
echo "  - Mystery: cat results/examples/creative-writing/mystery/*.md"
echo "  - Poetry: cat results/examples/creative-writing/poetry/*.json"
echo "  - Dialogue: cat results/examples/creative-writing/dialogue/*.md"
echo "  - Fantasy: cat results/examples/creative-writing/fantasy/*.json"
echo "  - Genres: ls results/examples/creative-writing/genres/"
echo "  - Style comparison: diff results/examples/creative-writing/style-test/minimalist/*.json results/examples/creative-writing/style-test/descriptive/*.json"
