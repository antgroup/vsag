commit=$(git rev-parse HEAD)
git ls-files | while read -r file; do
  time=$(git log -1 --pretty=%cd --date=iso -- "$file")
  if [ -n "$time" ]; then
    touch -d "$time" "$file"
  fi
done