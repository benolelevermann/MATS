import java.awt.image.BufferedImage;
import java.awt.image.DataBufferUShort;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

import loci.formats.FormatTools;
import loci.formats.ImageReader;

/** Read rectangular uint16 regions from Leica LOF/XLIF containers with Bio-Formats. */
public final class LeicaLofCropper {
    private LeicaLofCropper() {}

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            throw new IllegalArgumentException(
                    "Usage: LeicaLofCropper <input> [<output.tif> <x> <y> <width> <height>]...");
        }
        if ((args.length - 1) % 5 != 0) {
            throw new IllegalArgumentException("Each output requires output, x, y, width and height");
        }

        ImageReader reader = new ImageReader();
        try {
            reader.setId(args[0]);
            if (reader.getSeriesCount() != 1) {
                throw new IllegalArgumentException(
                        "Expected one overview series, found " + reader.getSeriesCount());
            }
            reader.setSeries(0);
            int sizeX = reader.getSizeX();
            int sizeY = reader.getSizeY();
            int sizeZ = reader.getSizeZ();
            int sizeC = reader.getSizeC();
            int sizeT = reader.getSizeT();
            int pixelType = reader.getPixelType();
            System.out.printf(
                    "INPUT=%s X=%d Y=%d Z=%d C=%d T=%d TYPE=%s LITTLE_ENDIAN=%s%n",
                    args[0], sizeX, sizeY, sizeZ, sizeC, sizeT,
                    FormatTools.getPixelTypeString(pixelType), reader.isLittleEndian());

            if (pixelType != FormatTools.UINT16 || sizeZ != 1 || sizeC != 1 || sizeT != 1) {
                throw new IllegalArgumentException(
                        "Expected one 2-D uint16 plane, got Z=" + sizeZ + " C=" + sizeC
                                + " T=" + sizeT + " type=" + FormatTools.getPixelTypeString(pixelType));
            }

            for (int index = 1; index < args.length; index += 5) {
                Path output = Path.of(args[index]);
                int x = Integer.parseInt(args[index + 1]);
                int y = Integer.parseInt(args[index + 2]);
                int width = Integer.parseInt(args[index + 3]);
                int height = Integer.parseInt(args[index + 4]);
                if (x < 0 || y < 0 || width <= 0 || height <= 0
                        || x + width > sizeX || y + height > sizeY) {
                    throw new IllegalArgumentException(
                            "Crop outside image bounds: x=" + x + " y=" + y
                                    + " width=" + width + " height=" + height);
                }

                byte[] raw = reader.openBytes(0, x, y, width, height);
                short[] pixels = new short[width * height];
                boolean littleEndian = reader.isLittleEndian();
                for (int pixel = 0, offset = 0; pixel < pixels.length; pixel++, offset += 2) {
                    int first = raw[offset] & 0xff;
                    int second = raw[offset + 1] & 0xff;
                    int value = littleEndian ? first | (second << 8) : (first << 8) | second;
                    pixels[pixel] = (short) value;
                }

                BufferedImage image = new BufferedImage(
                        width, height, BufferedImage.TYPE_USHORT_GRAY);
                short[] destination = ((DataBufferUShort) image.getRaster().getDataBuffer()).getData();
                System.arraycopy(pixels, 0, destination, 0, pixels.length);
                Files.createDirectories(output.toAbsolutePath().getParent());
                if (!ImageIO.write(image, "TIFF", output.toFile())) {
                    throw new IllegalStateException("No TIFF writer available for " + output);
                }
                System.out.printf(
                        "WROTE=%s X=%d Y=%d WIDTH=%d HEIGHT=%d BYTES=%d%n",
                        output, x, y, width, height, new File(output.toString()).length());
            }
        } finally {
            reader.close();
        }
    }
}
