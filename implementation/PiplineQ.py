import asyncio
import aiohttp
import aiofiles
import os
import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
from implementation import ColorCatImage, GrayCatImage, setup_logger



logger = setup_logger()
load_dotenv()
API_KEY = os.getenv("API_KEY")
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_DOWNLOAD_WORKERS = 3
NUM_CONV_WORKERS = 6
NUM_SAVE_WORKERS = 4


def convolution_worker(idx, breed, image, url):
    logger.debug(f"[ConvolutionWorker] Start processing idx={idx}, breed={breed}, image shape={image.shape}")
    if image.ndim == 3:
        cat = ColorCatImage(image, breed, url)
    else:
        cat = GrayCatImage(image, breed, url)
    result = {
        "original": cat,
        "cv_edges": cat.cv_edge_detection(),
        "my_edges": cat.my_edge_detection()
    }
    logger.debug(f"[ConvolutionWorker] Finished processing idx={idx}")
    return result


class PipelineState:
    def __init__(self):
        self.active_downloaders = NUM_DOWNLOAD_WORKERS
        self.active_processors = NUM_CONV_WORKERS

    def mark_downloader_done(self):
        self.active_downloaders -= 1
        logger.debug(f"[PipelineState] Downloader finished. Remaining: {self.active_downloaders}")
        return self.active_downloaders == 0

    def mark_processor_done(self):
        self.active_processors -= 1
        logger.debug(f"[PipelineState] Processor finished. Remaining: {self.active_processors}")
        return self.active_processors == 0


class Pipeline:

    def __init__(self, limit=5):
        self.limit = limit
        self.pipeline_state = PipelineState()
        self.url_queue = asyncio.Queue()
        self.download_queue = asyncio.Queue()
        self.processed_queue = asyncio.Queue()
        self.executor = ProcessPoolExecutor()

    # ---------- stage 1 ----------
    async def fetch_image_urls(self):
        logger.info("⬇️ Fetching image URLs...")
        url = "https://api.thecatapi.com/v1/images/search"
        headers = {"x-api-key": API_KEY}
        params = {"limit": self.limit, "has_breeds": 1}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as resp:
                data = await resp.json()

        logger.debug(f"[Fetcher] Received {len(data)} items from API")
        for idx, item in enumerate(data, start=1):
            breed = item["breeds"][0]["name"]
            url = item["url"]
            logger.debug(f"[Fetcher] Putting in queue: idx={idx}, breed={breed}, url={url}")
            await self.url_queue.put((idx, breed, url))

        for _ in range(NUM_DOWNLOAD_WORKERS):
            await self.url_queue.put(None)

    # ---------- stage 2 ----------
    async def download_worker(self, worker_id):
        processed_count = 0
        async with aiohttp.ClientSession() as session:
            while True:
                item = await self.url_queue.get()
                logger.debug(f"[Downloader {worker_id}] Queue size before get: {self.url_queue.qsize()}")
                if item is None:
                    self.url_queue.task_done()
                    if self.pipeline_state.mark_downloader_done():
                        for _ in range(NUM_CONV_WORKERS):
                            await self.download_queue.put(None)
                    break

                idx, breed, url = item
                logger.info(f"⬇️ [Downloader {worker_id}] Downloading image {idx}: {breed}")
                async with session.get(url, timeout=30) as resp:
                    data = await resp.read()
                logger.debug(f"[Downloader {worker_id}] Downloaded {len(data)} bytes for idx={idx}")

                img_arr = np.asarray(bytearray(data), dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
                logger.debug(f"[Downloader {worker_id}] Image shape after decode: {img.shape}")

                await self.download_queue.put((idx, breed, url, img))
                processed_count += 1
                self.url_queue.task_done()

        logger.info(f"✅ [Downloader {worker_id}] Processed {processed_count} images")
        return processed_count

    # ---------- stage 3 ----------
    async def convolution_worker_async(self, worker_id):
        processed_count = 0
        loop = asyncio.get_event_loop()

        while True:
            item = await self.download_queue.get()
            logger.debug(f"[Processor {worker_id}] Queue size before get: {self.download_queue.qsize()}")
            if item is None:
                self.download_queue.task_done()
                if self.pipeline_state.mark_processor_done():
                    await self.processed_queue.put(None)
                break

            idx, breed, url, img = item
            logger.info(f"⚙️ [Processor {worker_id}] Processing image {idx}")
            logger.debug(f"[Processor {worker_id}] Image shape: {img.shape}")

            result = await loop.run_in_executor(
                self.executor, convolution_worker, idx, breed, img, url
            )

            await self.processed_queue.put((idx, breed, result))
            processed_count += 1
            self.download_queue.task_done()

        logger.info(f"✅ [Processor {worker_id}] Processed {processed_count} images")
        return processed_count

    # ---------- stage 4 ----------
    async def save_worker(self):
        saved_count = 0
        while True:
            item = await self.processed_queue.get()
            if item is None:
                self.processed_queue.task_done()
                break

            idx, breed, results = item
            for method, cat in results.items():
                filename = f"{idx}_{breed.replace(' ', '_')}_{method}.png"
                path = os.path.join(SAVE_DIR, filename)
                _, png_data = cv2.imencode(".png", cat.image)
                logger.debug(f"[Saver] Saving {method} image for idx={idx}, shape={cat.image.shape}")

                async with aiofiles.open(path, "wb") as f:
                    await f.write(png_data.tobytes())

                saved_count += 1
                logger.info(f"💾 Saved: {filename}")

            self.processed_queue.task_done()

        logger.info(f"✅ [Saver] Saved {saved_count} images total")
        return saved_count

    # ---------- main run ----------
    async def run(self):
        logger.info("🚀 Starting pipeline...")
        start_time = time.perf_counter()

        logger.debug(f"[Pipeline] Limit={self.limit}, download_workers={NUM_DOWNLOAD_WORKERS}, "
                     f"conv_workers={NUM_CONV_WORKERS}, save_workers={NUM_SAVE_WORKERS}")

        fetch_task = asyncio.create_task(self.fetch_image_urls())

        download_tasks = [
            asyncio.create_task(self.download_worker(i))
            for i in range(NUM_DOWNLOAD_WORKERS)
        ]

        conv_tasks = [
            asyncio.create_task(self.convolution_worker_async(i))
            for i in range(NUM_CONV_WORKERS)
        ]

        save_tasks = [
            asyncio.create_task(self.save_worker())
            for _ in range(NUM_SAVE_WORKERS)
        ]

        await fetch_task
        download_results = await asyncio.gather(*download_tasks)
        logger.debug(f"[Pipeline] Download results: {download_results}")

        conv_results = await asyncio.gather(*conv_tasks)
        logger.debug(f"[Pipeline] Convolution results: {conv_results}")

        for _ in range(NUM_SAVE_WORKERS):
            await self.processed_queue.put(None)
        save_result = await asyncio.gather(*save_tasks)
        logger.debug(f"[Pipeline] Save results: {save_result}")

        self.executor.shutdown(wait=True)

        elapsed = time.perf_counter() - start_time
        total_downloaded = sum(download_results)
        total_processed = sum(conv_results)

        logger.info("=" * 60)
        logger.info("📊 PIPELINE STATISTICS:")
        logger.info(f"   Total downloaded: {total_downloaded}")
        logger.info(f"   Total processed: {total_processed}")
        logger.info(f"   Total saved: {save_result}")
        logger.info(f"   Total time: {elapsed:.2f} seconds")
        logger.info("🎉 Pipeline completed successfully!")
