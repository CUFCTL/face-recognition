# Contributing

Since this project tends to cycle through members quickly, this document should help new members figure out the project work-flow easily so they can focus on development.

## BLAS and LAPACK

Much of the code in this project depends on BLAS and LAPACK.

Documentation for BLAS and LAPACK consists mostly of the documentation for each function. For any given BLAS/LAPACK function, you will want to reference two documents:

1. The Fortran source file http://www.netlib.org/blas/ http://www.netlib.org/lapack/single/
2. The cblas/lapacke header http://www.netlib.org/blas/cblas.h http://www.netlib.org/lapack/lapacke.h

The Fortran source provides documentation for function parameters, and the C headers show how to order those arguments with the C interface.

### Installation

#### Ubuntu

    sudo apt-get install libblas-dev liblapacke-dev

#### Mac

- Download LAPACK 3.5.0 http://www.netlib.org/lapack/
- Download BLAS 3.5.0 http://www.netlib.org/blas/
- (10.10 - 10.11) Download gfortran 5.2 http://coudert.name/software/gfortran-5.2-Yosemite.dmg
- (10.7 - 10.9) Download gfortran 4.8.2 http://coudert.name/software/gfortran-4.8.2-MountainLion.dmg

```
# in BLAS directory
make
sudo cp blas-LINUX.a /usr/local/lib/libblas.a

# in LAPACK directory
cp make.inc.example make.inc
# set BLASLIB in make.inc line 68 equal to ‘/usr/local/lib/libblas.a’
make
sudo cp liblapack.a /usr/local/lib
```

## Git Basics

Git is a _very_ powerful version control system with a lot of features, so this guide will just give you the main commands to know for everyday use. You should try to learn about Git a little at a time as you go, but in the meantime, if you follow this guide you shouldn't have any trouble.

### Setup

If you haven't used Git before, you will need to install git. For Linux distributions, git can be installed through the package manager. For Windows and Mac, you should either set up a dual boot with Linux or use [Git Bash](https://git-scm.com/downloads). __Do not use a GUI client for Git or Github!__ Unless you are very experienced with git, you will very likely cause problems if you use any of the GUI clients for git.

With git installed, configure your username:

    git config --global user.name [name]
    git config --global user.email [email-address]

To get started, clone this repository in your local workspace:

    git clone https://github.com/CUFCTFACE/face-recognition.git
    cd face-recognition
    git remote add origin https://github.com/CUFCTFACE/face-recognition.git
    git push -u origin master

The last command creates an alias called "origin" which will help you pull from and push to the main repo.

### Developing

As you make changes in your workspace, you can track them with `git status`.

You can also use `git diff` to show a diff of the files you have changed (use `git diff --cached` for files that are staged).

When you're ready to commit your changes to your local repo, use `git add [files]` to stage the changed files and `git commit -m [message]` to commit them. You have to have a message for every commit. Use `git status` when you add files to make sure they are staged.

If the files you changed are already being tracked, you can use `git commit -am [message]` to stage and commit at the same time.

If you screw up a commit, use `git reset HEAD^` to undo the commit (you can add more `^`'s to undo multiple commits). Your files will be the same, just with the uncommitted changes. Try to catch mistakes before you push them to the main repo, because it's a pain to undo them after that!

Try to make your commits clean and modular, e.g. make changes to add one feature or fix one bug, make sure the changes work, then commit them. That will make your commit message easy to write. Don't just change a bunch of stuff all over the place in one commit! That will make it hard for people in the future to figure out what you changed.

### Pulling and pushing

When you join the project, you will be given push rights to this repository. That means you have to be careful about pushing your changes, because everyone else can push their changes too, and if we don't push responsibly then there will be conflicts. With that in mind, try to stick to the following work-flow:

1. Before you develop, pull any new commits from Github:
```
git pull
```
2. Make sure no one else is working on the same code as you, but if you are working with someone else on the same code, you should probably coordinate in person.
3. Develop!
4. Before you commit your changes, you might want to `pull` again just in case someone pushed more commits while you were working.
5. Push your commits to Github:
```
git push
```

### In Case of Emergency

If you commit new changes and then try to pull, git won't let you pull! And if you try to push but the main repo has new commits that you didn't pull, git won't let you push! At least not with the commands in this guide, so don't try to force git in these situations because you'll make a mess! Follow these steps instead:

1. Undo your commits with `git reset HEAD^` until you're on the same page as the main repo.
2. Pull
3. Commit your changes
4. Push

### Branching

Sometimes you might want to experiment with a new feature, but you don't want to add it to the project yet. Maybe other people are working on the same feature so you need to collaborate. In this case, you can create a separate branch to version control your experimental feature. The main repo that you commit to is itself a branch called `master`; to make a new branch and use it, follow these steps:

1. Create the branch and switch to it:
```
git checkout -b [branch-name]
```
2. Develop, commit changes to your branch with `git commit`.
3. Push your branch to Github with `git push`.
