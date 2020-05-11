# Intro
git: system for version control  
github: website to upload your directory

.md (markdown file):

\#\# This is the header  
Regular text  
\* bullet 1  
\[Link to Google\](http://www.google.com)  

fork-&gt;modify-&gt;pull request (does not sync with the upstream repo)

# Init by clone
we can first init the repo on github, and then clone it to our local
machine.

open git bash:  
```shell
cd \~/Desktop/
git clone *url*
cd *test-repo* (now we see the master branch, test-repo is directory
name)
git remote -v
git remote add origin *url* (can skip the step if origin already exist)
touch new.md (create a file on windows with non standard extension)
git status
git add new.md (add one file)
git add . (add all files once)
git commit -m “edit readme and add new.md” (remember to add the message)
git log
git push origin master
```

# Init by fork
fork other people’s repo and there are some new commit in that repo:  
```shell
git remote add upstream *url*
git fetch upstream
git merge upstream/master (new update is in the branch upstream/master)
git push origin master
```

# Sync
```
mkdir test1
cd test1
git init
git remote add origin <http://github.com/username/test1.git>
cp ../test.txt test.txt
git status
git add .
git commit -m “msg”
git push origin master **--此时出现error显示fail to push some refs to …
updates were rejected because the remote contains work that you do not
have locally**
git pull origin master
git push origin master --此时可以成功push
```
从远端拿东西最好用git fetch + git merge。如果直接使用git pull会覆盖本地有但是远端没有的文件。  


# Branch
```
git checkout -b newbranch    #新建branch，并且进入到new branch
git checkout master          #切换回到master
git merge newbranch          #这里是分支名字，将new branch内容合并到master branch
git branch -d newbranch      #当前分支已经没用了，记得删除，除非还需要用到
git branch                   #查看当前所在分支
git branch -A/--all          #查看所有分支
```

# Roll Back
```
git log                                 #查看版本
git reset --hard <commit_id>            #回退到指定版本
git push origin HEAD --force            #清空这个commitid之后所有已经提交了的commit
git revert <commit-id>                  #剔除某次提交，其后的commit不受到影响
```

