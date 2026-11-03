import subprocess
result = subprocess.run(
  ['git', 'status'],
  capture_output=True, 
  text=True, 
  cwd='C:\\Users\\AORUS\\work\\oss-camp\\vsag-wt-2872'
)
with open('git_output.txt', 'w') as f:
    f.write('returncode: ' + str(result.returncode) + '\n')
    f.write('stdout: ' + result.stdout + '\n')
    f.write('stderr: ' + result.stderr + '\n')
print('done')