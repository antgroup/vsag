import subprocess
result = subprocess.run(['git', 'diff', '--stat'], capture_output=True, text=True, cwd='C:\\Users\\AORUS\\work\\oss-camp\\vsag-wt-2872')
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)