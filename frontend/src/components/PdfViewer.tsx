import { useEffect, useRef, useState } from 'react'
import * as pdfjs from 'pdfjs-dist'
// Vite serves the worker file as an asset URL.
// @ts-expect-error -- ?url is a Vite import suffix
import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

import { pdfUrl } from '../api'
import type { Span } from '../types'

pdfjs.GlobalWorkerOptions.workerSrc = workerSrc

// Cache document load promises so jumping between spans in the same PDF doesn't refetch.
const DOC_CACHE = new Map<string, Promise<pdfjs.PDFDocumentProxy>>()
function loadDoc(url: string): Promise<pdfjs.PDFDocumentProxy> {
  let p = DOC_CACHE.get(url)
  if (!p) {
    p = pdfjs.getDocument(url).promise
    DOC_CACHE.set(url, p)
  }
  return p
}

export function PdfViewer({
  sessionId,
  span,
}: {
  sessionId: string
  span: Span | null
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const renderTaskRef = useRef<{ cancel: () => void } | null>(null)
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null)

  const docId = span?.doc_id ?? null
  const page = span?.page ?? null

  useEffect(() => {
    if (!docId || !page) {
      setDims(null)
      return
    }
    let cancelled = false
    ;(async () => {
      const doc = await loadDoc(pdfUrl(sessionId, docId))
      if (cancelled) return
      const pageProxy = await doc.getPage(page)
      if (cancelled) return

      const container = containerRef.current
      const targetWidth = Math.max(320, (container?.clientWidth ?? 800) - 24)
      const baseViewport = pageProxy.getViewport({ scale: 1 })
      const scale = targetWidth / baseViewport.width
      const viewport = pageProxy.getViewport({ scale })

      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      canvas.width = Math.floor(viewport.width)
      canvas.height = Math.floor(viewport.height)
      canvas.style.width = `${viewport.width}px`
      canvas.style.height = `${viewport.height}px`

      if (renderTaskRef.current) {
        try {
          renderTaskRef.current.cancel()
        } catch {
          // ignore
        }
      }

      const task = pageProxy.render({ canvasContext: ctx, viewport })
      renderTaskRef.current = task
      try {
        await task.promise
      } catch (e) {
        const name = (e as { name?: string } | null)?.name
        if (name !== 'RenderingCancelledException') throw e
        return
      }
      if (!cancelled) {
        setDims({ w: viewport.width, h: viewport.height })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [sessionId, docId, page])

  useEffect(() => {
    if (!span || !dims || !containerRef.current) return
    const top = span.bbox[1] * dims.h
    containerRef.current.scrollTo({
      top: Math.max(0, top - 60),
      behavior: 'smooth',
    })
  }, [span?.span_id, dims])

  if (!span) {
    return (
      <div className="h-full flex items-center justify-center text-slate-500 text-sm p-6 text-center">
        Click any span on the left to see it highlighted in the source PDF.
      </div>
    )
  }

  return (
    <div ref={containerRef} className="overflow-auto h-full p-3 bg-slate-100">
      <div className="relative inline-block shadow bg-white">
        <canvas ref={canvasRef} />
        {dims && (
          <div
            className="absolute pointer-events-none border-2 border-yellow-400 bg-yellow-300/30"
            style={{
              left: span.bbox[0] * dims.w,
              top: span.bbox[1] * dims.h,
              width: (span.bbox[2] - span.bbox[0]) * dims.w,
              height: (span.bbox[3] - span.bbox[1]) * dims.h,
            }}
          />
        )}
      </div>
      <div className="text-xs text-slate-500 mt-2 max-w-[900px] break-words">
        <span className="font-medium">p.{span.page}</span> · {span.source} ·{' '}
        {span.text}
      </div>
    </div>
  )
}
