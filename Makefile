mypy:
	mypy --ignore-missing-imports --follow-imports=skip examples/
clean:
	find . -name *.out -or -name *.log -or -name *.swp -or -name *.swo -or -name .DS_Store -or -name .swp -or -name *.pyc | xargs -n 1 rm
