"use client";
import dynamic from "next/dynamic";
import LoadingMap from "./map/LoadingMap";

const Map = dynamic(() => import("./map/MapComponent"), {
  ssr: false,
  loading: LoadingMap,
});

const FinalLoadoutThingy: React.FC = () => {
  return (
    <section className="w-full h-[55vh] flex flex-col">
      <Map />
    </section>
  );
};

export default FinalLoadoutThingy;
