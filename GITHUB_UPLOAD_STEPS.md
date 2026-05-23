# GitHub Upload Steps

Use these steps after downloading this prepared project folder.

## Option A: Upload using the GitHub website

This is the easiest method.

1. Open your repository: `https://github.com/arianasa01/Spotify-hit-prediction`
2. Click **Add file**.
3. Click **Upload files**.
4. Open the prepared folder on your laptop.
5. Drag these files and folders into GitHub:

   ```text
   README.md
   requirements.txt
   .gitignore
   data/
   src/
   reports/
   notebooks/
   images/
   outputs/
   ```

6. Scroll down to **Commit changes**.
7. In the commit message, write:

   ```text
   Add full project structure and reproducible workflow
   ```

8. Click **Commit changes**.

## Important

Do not upload the raw CSV file.

The `.gitignore` file is already set to ignore:

```text
data/raw/*.csv
data/processed/*.csv
```

This means the large dataset will stay on your laptop and will not be uploaded by mistake.
