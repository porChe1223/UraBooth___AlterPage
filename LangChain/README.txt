brew install --cask google-cloud-sdk
echo 'source "$(brew --prefix)/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc"' >> ~/.zshrc
source ~/.zshrc
gcloud init
gcloud auth application-default login
環境変数でgoogle cloudの.jsonを指定