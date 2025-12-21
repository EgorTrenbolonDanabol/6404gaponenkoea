import asyncio
import aiohttp
import aiofiles
import os
import cv2
import numpy as np
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from lab4.implementation.ColorCatImage import ColorCatImage
from lab4.implementation.GreyCatImage import GreyCatImage
from rich.logging import RichHandler
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True)]
)

load_dotenv()
API_KEY = os.getenv("API_KEY")
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_DOWNLOAD_WORKERS = 6
NUM_CONV_WORKERS = 3


def convolution_worker(idx, breed, image, url):
    if image.ndim == 3:
        cat = ColorCatImage(image, breed, url)
    else:
        cat = GreyCatImage(image, breed, url)
    return {
        "original": cat,
        "cv_edges": cat.edge_detection_cv(),
        "cv_corners": cat.corner_detection_cv()
    }


class PipelineState:
    def __init__(self):
        self.active_downloaders = NUM_DOWNLOAD_WORKERS
        self.active_processors = NUM_CONV_WORKERS

    def mark_downloader_done(self):
        self.active_downloaders -= 1
        return self.active_downloaders == 0

    def mark_processor_done(self):
        self.active_processors -= 1
        return self.active_processors == 0


async def fetch_image_urls(limit, url_queue: asyncio.Queue):
    url = "https://api.thecatapi.com/v1/images/search"
    headers = {"x-api-key": API_KEY}
    params = {"limit": limit, "has_breeds": 1}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as resp:
            data = await resp.json()

    for idx, item in enumerate(data, start=1):
        await url_queue.put((idx, item["breeds"][0]["name"], item["url"]))

    for _ in range(NUM_DOWNLOAD_WORKERS):
        await url_queue.put(None)


async def download_worker(worker_id, url_queue, download_queue, pipeline_state):
    processed_count = 0
    async with aiohttp.ClientSession() as session:
        while True:
            item = await url_queue.get()
            if item is None:
                url_queue.task_done()
                if pipeline_state.mark_downloader_done():
                    for _ in range(NUM_CONV_WORKERS):
                        await download_queue.put(None)
                break

            idx, breed, url = item
            logging.info(f"⬇️ [Downloader {worker_id}] Downloading image {idx}: {breed}")
            async with session.get(url, timeout=30) as resp:
                data = await resp.read()

            img_arr = np.asarray(bytearray(data), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)

            await download_queue.put((idx, breed, url, img))
            processed_count += 1
            url_queue.task_done()

    logging.info(f"✅ [Downloader {worker_id}] Processed {processed_count} images")
    return processed_count


async def convolution_worker_async(worker_id, download_queue, processed_queue, executor, pipeline_state):
    processed_count = 0
    loop = asyncio.get_event_loop()

    while True:
        item = await download_queue.get()
        if item is None:
            download_queue.task_done()
            if pipeline_state.mark_processor_done():
                await processed_queue.put(None)
            break

        idx, breed, url, img = item
        logging.info(f"⚙️ [Processor {worker_id}] Processing image {idx}")
        result = await loop.run_in_executor(executor, convolution_worker, idx, breed, img, url)
        await processed_queue.put((idx, breed, result))
        processed_count += 1
        download_queue.task_done()

    logging.info(f"✅ [Processor {worker_id}] Processed {processed_count} images")
    return processed_count


async def save_worker(processed_queue):
    saved_count = 0
    while True:
        item = await processed_queue.get()
        if item is None:
            processed_queue.task_done()
            break

        idx, breed, results = item
        for method, cat in results.items():
            filename = f"{idx}_{breed.replace(' ', '_')}_{method}.png"
            path = os.path.join(SAVE_DIR, filename)
            _, png_data = cv2.imencode(".png", cat.image)
            async with aiofiles.open(path, "wb") as f:
                await f.write(png_data.tobytes())
            saved_count += 1
            logging.info(f"💾 Saved: {filename}")

        processed_queue.task_done()

    logging.info(f"✅ [Saver] Saved {saved_count} images total")
    return saved_count


async def main(limit=5):
    pipeline_state = PipelineState()
    url_queue = asyncio.Queue()
    download_queue = asyncio.Queue()
    processed_queue = asyncio.Queue()
    executor = ProcessPoolExecutor()

    logging.info("🚀 Starting pipeline...")
    start_time = time.perf_counter()

    # Этап 1: получение URL
    fetch_task = asyncio.create_task(fetch_image_urls(limit, url_queue))

    # Этап 2: скачивание
    download_tasks = []
    for i in range(NUM_DOWNLOAD_WORKERS):
        task = asyncio.create_task(download_worker(i, url_queue, download_queue, pipeline_state))
        download_tasks.append(task)

    # Этап 3: обработка изображений
    conv_tasks = []
    for i in range(NUM_CONV_WORKERS):
        task = asyncio.create_task(
            convolution_worker_async(i, download_queue, processed_queue, executor, pipeline_state)
        )
        conv_tasks.append(task)

    # Этап 4: сохранение
    save_task = asyncio.create_task(save_worker(processed_queue))

    await fetch_task
    download_results = await asyncio.gather(*download_tasks)
    conv_results = await asyncio.gather(*conv_tasks)
    save_result = await save_task

    executor.shutdown(wait=True)

    elapsed = time.perf_counter() - start_time
    total_downloaded = sum(download_results)
    total_processed = sum(conv_results)

    logging.info("=" * 60)
    logging.info("📊 PIPELINE STATISTICS:")
    logging.info(f"   Total downloaded: {total_downloaded}")
    logging.info(f"   Total processed: {total_processed}")
    logging.info(f"   Total saved: {save_result}")
    logging.info(f"   Total time: {elapsed:.2f} seconds")
    logging.info("🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(main(limit=20))
