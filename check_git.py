import os
base_git = 'C:\\Users\\AORUS\\work\\oss-camp\\vsag-base\\.git\\worktrees\\vsag-wt-2872'
print('exists:', os.path.exists(base_git))
if os.path.exists(base_git):
    print('contents:', os.listdir(base_git))