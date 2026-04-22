#!/bin/bash


if [ ! -z "$(git status --untracked-files=no --porcelain)" ]; then
    echo "You have changes that have yet to be committed."
    echo "Aborting."
    exit 1
fi


if [ -d "dist" ]; then
  rm -rf ./dist
fi

VERSION=$(uv version --short)
echo "Current version is ${VERSION}"
read -p "Set new version (default is the same): " NEW_VERSION
NEW_VERSION=${NEW_VERSION:-${VERSION}}
uv version ${NEW_VERSION}
uv build

git add --all
git commit -m "Release PBSim Commons Version: ${NEW_VERSION}"
git push

TOKEN=${PYPI_TOKEN-"nope"}
if [[ ${TOKEN} == "nope" ]]; then
    read -s -p "Token for pypi: " TOKEN
fi

uv publish --token=${TOKEN}
