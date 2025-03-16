"use client";

import { useEffect, useRef, useState } from 'react';
import 'leaflet/dist/leaflet.css';
import Image from 'next/image';

const MapComponent: React.FC = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const [location, setLocation] = useState<[number, number] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    import('leaflet').then(L => {
    
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const lat = position.coords.latitude;
          const lng = position.coords.longitude;
          setLocation([lat, lng]);
          
          if (mapRef.current && !mapRef.current.innerHTML) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            delete (L.Icon.Default.prototype as any)._getIconUrl;
            L.Icon.Default.mergeOptions({
              iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
              iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
              shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
            });

            const map = L.map(mapRef.current).setView([lat, lng], 15);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
              attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);

            const autoIcon = L.icon({
              iconUrl: '/auto.png',
              iconSize: [40, 40],
              iconAnchor: [20, 20],
              popupAnchor: [0, -20]
            });

            const marker = L.marker([lat, lng], { icon: autoIcon }).addTo(map);
            marker.bindPopup("Your Auto").openPopup();

            L.circle([lat, lng], {
              color: '#FBBF24', // Changed to yellow-400 equivalent
              fillColor: '#FBBF24', // Changed to yellow-400 equivalent
              fillOpacity: 0.1,
              radius: 100
            }).addTo(map);
          }
        },
        (error) => {
          console.log("Error getting location:", error);
          setError("Location services denied or unavailable");
        }
      );
    } else {
      console.log("Geolocation is not supported by this browser.");
      setError("Location services not supported by your browser");
    }
    }).catch(err => {
      console.error("Failed to load map:", err);
      setError("Failed to load map");
    });
  }, []);

  return (
    <>
      {error ? (
        <div className="w-full h-full gap-3 flex flex-col items-center justify-center bg-gray-900">
          <Image 
            src="/pepecry.png" 
            alt="Location Error" 
            className="w-32 h-32 mb-4"
            width={128}
            height={128}
            onError={(e) => {
              e.currentTarget.src = "https://placehold.co/200x200?text=Location+Error";
            }}
          />
          <p className="text-yellow-400 font-bold text-center mt-2">{error}</p>
          <p className="text-gray-400 mt-2 text-center w-2/3">Please enable location services and refresh...</p>
        </div>
      ) : (
        <>
          <div className="w-full h-full" ref={mapRef}></div>
          {!location && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-70">
              <div className="text-center">
                <svg className="animate-spin h-10 w-10 text-yellow-400 mx-auto mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <p className="text-gray-300">Getting your location...</p>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
};

export default MapComponent;