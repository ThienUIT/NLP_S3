PORT ?= 8000

.PHONY: help serve main run run2 run3 rungoogle rungoogle1 relation \
        open open2 opengoogle opengoogle1 openrelation pptx openpptx deploy clean

help:
	@echo "=== NLP (S3 - Semantic Signal Separation) — thu muc NLP/ ==="
	@echo "  make main        - chay server + mo NLP/main/main.html      (SLIDE CHINH - THUYET TRINH)"
	@echo "  make run         - chay server + mo NLP/temp/slides.html    (ban de HOC day du)"
	@echo "  make run3        - chay server + mo NLP/temp/slides_3.html  (HOC SAU Phan 3)"
	@echo "  make rungoogle   - chay server + mo NLP/temp/google/google_slides.html"
	@echo "  make rungoogle1  - chay server + mo NLP/temp/google/google_slides_1.html"
	@echo "  make open2       - mo NLP/main/main.html    truc tiep bang trinh duyet"
	@echo "  make open        - mo NLP/temp/slides.html  truc tiep bang trinh duyet"
	@echo "  make pptx        - dung lai NLP/main/main.pptx tu build_pptx.py"
	@echo "  make openpptx    - mo NLP/main/main.pptx bang PowerPoint/Keynote"
	@echo ""
	@echo "=== relation (LeWorldModel) — thu muc relation/ ==="
	@echo "  make relation    - chay server + mo relation/main/index.html"
	@echo "  make openrelation- mo relation/main/index.html truc tiep bang trinh duyet"
	@echo ""
	@echo "=== chung ==="
	@echo "  make serve       - chay local server tai http://localhost:$(PORT)/ (PORT=xxxx de doi cong)"
	@echo "  make deploy      - deploy len Vercel production (/ /nlp /1 /2 /3 /g... /relation)"
	@echo "  make clean       - xoa file tam (.vercel/)"

serve:
	python3 -m http.server $(PORT)

# ---------- NLP ----------

main:
	@echo "Mo http://localhost:$(PORT)/NLP/main/main.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/NLP/main/main.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

# alias: run2 = main (giu tuong thich cu)
run2: main

run:
	@echo "Mo http://localhost:$(PORT)/NLP/temp/slides.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/NLP/temp/slides.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

run3:
	@echo "Mo http://localhost:$(PORT)/NLP/temp/slides_3.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/NLP/temp/slides_3.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

rungoogle:
	@echo "Mo http://localhost:$(PORT)/NLP/temp/google/google_slides.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/NLP/temp/google/google_slides.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

rungoogle1:
	@echo "Mo http://localhost:$(PORT)/NLP/temp/google/google_slides_1.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/NLP/temp/google/google_slides_1.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

open:
	open NLP/temp/slides.html

open2:
	open NLP/main/main.html

opengoogle:
	open NLP/temp/google/google_slides.html

opengoogle1:
	open NLP/temp/google/google_slides_1.html

pptx:
	cd NLP/main && python3 build_pptx.py

openpptx:
	open NLP/main/main.pptx

# ---------- relation ----------

relation:
	@echo "Mo http://localhost:$(PORT)/relation/main/index.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/relation/main/index.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

openrelation:
	open relation/main/index.html

# ---------- chung ----------

deploy:
	vercel deploy --prod

clean:
	rm -rf .vercel
