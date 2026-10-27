# Pubblicazione su GitHub (PowerShell)

Il repository locale ha una storia Git nuova e separata dal workspace originale.
Nessun dataset o simulatore è stato copiato. Nessun remote è configurato e
nessun contenuto viene pubblicato automaticamente.

## Prima della pubblicazione

1. Verificare con gli eventuali titolari i diritti di pubblicazione del codice.
   La mancata distribuzione di dati/simulatore non sostituisce questa verifica.
2. Leggere `CHAPTER_ALIGNMENT.md`. Non descrivere il repository come archivio
   capace di riprodurre tutte le tabelle/figure storiche.
3. Decidere l'eventuale licenza con l'istituzione: non è stata applicata una
   licenza senza autorizzazione. Se aggiunta, aggiornare anche l'allowlist.
4. Non aggiungere dataset, manifest reali, checkpoint o notebook eseguiti.

## Controllo e commit iniziale

Aprire PowerShell nella cartella `radar-tracklet-thesis` e lanciare:

```powershell
python tools/check_release.py
git status --short
git add -- .gitignore .gitattributes README.md pyproject.toml src notebooks tests docs tools
git diff --cached --stat
git diff --cached
git commit -m "Add code-only thesis review companion"
git branch -M main
```

Se Git richiede l'identità, configurare nome/e-mail appropriati con
`git config user.name` e `git config user.email` **nel solo repository**.
Controllare che l'e-mail sia adatta a essere resa pubblica nella storia Git.

## Creazione e upload dal sito GitHub

Creare sul proprio account un repository **Public** chiamato per esempio
`radar-tracklet-thesis`, senza README, licenza o gitignore automatici.
Quindi sostituire `TUO_ACCOUNT`:

```powershell
git remote add origin https://github.com/TUO_ACCOUNT/radar-tracklet-thesis.git
git remote -v
git push -u origin main
```

Non usare `--force`. Nessun token va scritto nei notebook o nei file del repo.
Autenticarsi con il meccanismo già configurato per Git/GitHub.

Il link da condividere sarà:

```text
https://github.com/TUO_ACCOUNT/radar-tracklet-thesis
```

Per un riferimento stabile da tesi/revisori creare un tag dopo il controllo:

```powershell
git tag -a thesis-review-v1 -m "Code companion for thesis review"
git push origin thesis-review-v1
```

Usare il link al tag `.../tree/thesis-review-v1`, oppure al commit esatto.
Questo repository non contiene il capitolo integrale: va condiviso separatamente.

