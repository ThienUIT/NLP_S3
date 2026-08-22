PORT ?= 8000

.PHONY: help serve main run run2 run3 rungoogle rungoogle1 open open2 opengoogle opengoogle1 pptx openpptx deploy clean demo

help:
	@echo "Targets:"
	@echo "  make serve       - chay local server tai http://localhost:$(PORT)/ (PORT=xxxx de doi cong)"
	@echo "  make main        - chay server + mo main/main.html      (SLIDE CHINH - THUYET TRINH)"
	@echo "  make run         - chay server + mo temp/slides.html    (ban de HOC day du)"
	@echo "  make run3        - chay server + mo temp/slides_3.html  (HOC SAU Phan 3)"
	@echo "  make rungoogle   - chay server + mo temp/google/google_slides.html"
	@echo "  make rungoogle1  - chay server + mo temp/google/google_slides_1.html"
	@echo "  make open        - mo temp/slides.html   truc tiep bang trinh duyet"
	@echo "  make open2       - mo main/main.html     truc tiep bang trinh duyet"
	@echo "  make pptx        - dung lai main/main.pptx (PowerPoint) tu build_pptx.py"
	@echo "  make openpptx    - mo main/main.pptx bang PowerPoint/Keynote"
	@echo "  make deploy      - deploy len Vercel production (cap nhat / /1 /2 /3 /g...)"
	@echo "  make clean       - xoa file tam (.vercel/)"
	@echo "  make demo        - chay Streamlit demo phan tich truc topic S3 (demo/app.py)"

serve:
	python3 -m http.server $(PORT)

main:
	@echo "Mo http://localhost:$(PORT)/main/main.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/main/main.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

# alias: run2 = main (giu tuong thich cu)
run2: main

run:
	@echo "Mo http://localhost:$(PORT)/temp/slides.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/temp/slides.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

run3:
	@echo "Mo http://localhost:$(PORT)/temp/slides_3.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/temp/slides_3.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

rungoogle:
	@echo "Mo http://localhost:$(PORT)/temp/google/google_slides.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/temp/google/google_slides.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

rungoogle1:
	@echo "Mo http://localhost:$(PORT)/temp/google/google_slides_1.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/temp/google/google_slides_1.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

open:
	open main/main.html

open2:
	open temp/slides.html

opengoogle:
	open temp/google/google_slides.html

opengoogle1:
	open temp/google/google_slides_1.html

pptx:
	cd main && python3 build_pptx.py

openpptx:
	open main/main.pptx

demo:
	.venv/Scripts/python.exe -m streamlit run demo/app.py

deploy:
	vercel deploy --prod

clean:
	rm -rf .vercel
