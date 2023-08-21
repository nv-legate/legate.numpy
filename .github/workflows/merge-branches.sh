#!/bin/bash

main() {
  REMOTES="$@";
  if [ -z "$REMOTES" ]; then
    REMOTES=$(git remote);
  fi
  REMOTES=$(echo "$REMOTES" | xargs -n1 echo)
  CLB=$(git rev-parse --abbrev-ref HEAD);
  echo "$REMOTES" | while read REMOTE; do
    git remote update $REMOTE
    git branch -r \
    | git branch -r | awk 'BEGIN { FS = "/" };/'"$REMOTE"'/{print $2}'  \
    | while read BRANCH; do
      if [[ $BRANCH == !(main|branch-*|gh-pages) ]]; then
        echo "Skipping branch $BRANCH because it does not match the (main|branch-*) pattern.";
        continue;
      fi
      # first, delete the local branch if there is one
      git branch -D $BRANCH 2>/dev/null || true
      # checkout the branch tracking from origin or fail if there isn't one yet
      git checkout --track origin/$BRANCH 2>/dev/null || true
      # reset the branch, or fail if the branch is not checked out
      git reset --hard origin/$BRANCH 2>/dev/null || true
      ARB="refs/remotes/$REMOTE/$BRANCH";
      ALB="refs/heads/$BRANCH";
      NBEHIND=$(( $(git rev-list --count $ALB..$ARB 2>/dev/null || echo "-1") ));
      NAHEAD=$(( $(git rev-list --count $ARB..$ALB 2>/dev/null || true) ));
      if [ "$NBEHIND" -gt 0 ]; then
        if [ "$NAHEAD" -gt 0 ]; then
          echo " branch $BRANCH is $NAHEAD commit(s) ahead of $REMOTE/$BRANCH.  Public branches cannot contain internal commits.";
          exit 1;
        else
          echo " branch $BRANCH was $NBEHIND commit(s) behind of $REMOTE/$BRANCH. resetting local branch to remote";
          git reset --hard $REMOTE/$BRANCH >/dev/null;
          git push origin $BRANCH
        fi
      elif [ "$NBEHIND" -eq -1 ]; then
          echo " branch $BRANCH does not exist yet. Creating a new branch to track remote";
          git branch -f $BRANCH -t $ARB >/dev/null;
          git push origin $BRANCH
      else
          echo "Nothing to be done for branch $BRANCH"
      fi
    done
  done
}

main $@

