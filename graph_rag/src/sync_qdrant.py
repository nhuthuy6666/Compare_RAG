from ingest import main


## Wrapper tương thích cũ: giữ lệnh `sync_qdrant.py` nhưng chuyển thẳng sang flow ingest Qdrant-only mới.
if __name__ == "__main__":
    main()
